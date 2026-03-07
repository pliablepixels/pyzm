"""Low-level HTTP wrapper for the ZoneMinder API.

Wraps ``requests.Session`` with automatic authentication, token refresh,
401 relogin, content-type detection, and SSL toggle.  All configuration
comes from a :class:`pyzm.models.config.ZMClientConfig` Pydantic model.

No global state -- everything is instance-based.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from pyzm.models.config import ZMClientConfig
from pyzm.zm.auth import AuthManager

logger = logging.getLogger("pyzm.zm")


class ZMAPI:
    """Low-level ZoneMinder API client.

    Parameters
    ----------
    config:
        Connection settings expressed as a ``ZMClientConfig`` model.
    """

    def __init__(self, config: ZMClientConfig) -> None:
        self.config = config
        self._api_url = config.api_url.rstrip("/")
        self._portal_url = (config.portal_url or "").rstrip("/")

        # Build session
        self._session = requests.Session()

        if config.basic_auth_user:
            logger.debug("Configuring basic auth for session")
            password = (
                config.basic_auth_password.get_secret_value()
                if config.basic_auth_password
                else ""
            )
            self._session.auth = (config.basic_auth_user, password)

        if not config.verify_ssl:
            self._session.verify = False
            logger.debug("SSL certificate verification disabled")
            # Suppress only the specific urllib3 warning
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Build auth manager and login
        password_plain = (
            config.password.get_secret_value() if config.password else None
        )
        self._auth = AuthManager(
            session=self._session,
            api_url=self._api_url,
            user=config.user,
            password=password_plain,
        )
        self._auth.login()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def portal_url(self) -> str:
        return self._portal_url

    @property
    def api_version(self) -> str | None:
        return self._auth.api_version

    @property
    def zm_version(self) -> str | None:
        return self._auth.zm_version

    @property
    def session(self) -> requests.Session:
        return self._session

    @property
    def auth(self) -> AuthManager:
        return self._auth

    # ------------------------------------------------------------------
    # Core request method
    # ------------------------------------------------------------------

    def request(
        self,
        url: str,
        method: str = "get",
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        reauth: bool = True,
    ) -> Any:
        """Make an HTTP request with automatic auth injection.

        Parameters
        ----------
        url:
            Full URL to request.
        method:
            HTTP method (``get``, ``post``, ``put``, ``delete``).
        params:
            Query parameters.
        payload:
            Form data for POST/PUT.
        reauth:
            If ``True`` (default), a 401 response triggers a single
            relogin-and-retry cycle.

        Returns
        -------
        Any
            Parsed JSON (``dict``), a ``requests.Response`` for image
            content, or ``None`` for empty DELETE responses.

        Raises
        ------
        requests.HTTPError
            On non-recoverable HTTP errors.
        ValueError
            ``"BAD_IMAGE"`` when ZM returns a zero-byte or 404 image.
            ``"RELOGIN"`` when a non-JSON, non-image response is received
            (usually means the token expired mid-stream).
        """
        self._auth.refresh_if_needed()

        if params is None:
            params = {}
        url, params = self._auth.apply_auth(url, params)

        method = method.lower()
        logger.debug("HTTP %s %s params=%s", method.upper(), url, params)

        try:
            resp = self._dispatch(method, url, params, payload)
            resp.raise_for_status()
            return self._parse_response(resp, method)

        except requests.HTTPError as exc:
            return self._handle_http_error(exc, url, method, params, payload, reauth)

        except ValueError as exc:
            return self._handle_value_error(exc, url, method, params, payload, reauth)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: dict[str, str] | None = None) -> Any:
        """GET ``{api_url}/{endpoint}``."""
        return self.request(f"{self._api_url}/{endpoint}", params=params)

    def post(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """POST ``{api_url}/{endpoint}``."""
        return self.request(
            f"{self._api_url}/{endpoint}", method="post", payload=data,
        )

    def put(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """PUT ``{api_url}/{endpoint}``."""
        return self.request(
            f"{self._api_url}/{endpoint}", method="put", payload=data,
        )

    def delete(self, endpoint: str) -> Any:
        """DELETE ``{api_url}/{endpoint}``."""
        return self.request(f"{self._api_url}/{endpoint}", method="delete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        method: str,
        url: str,
        params: dict[str, str],
        payload: dict[str, Any] | None,
    ) -> requests.Response:
        """Dispatch the actual HTTP call."""
        match method:
            case "get":
                return self._session.get(url, params=params, timeout=self.config.timeout)
            case "post":
                return self._session.post(
                    url, data=payload, params=params, timeout=self.config.timeout,
                )
            case "put":
                return self._session.put(
                    url, data=payload, params=params, timeout=self.config.timeout,
                )
            case "delete":
                return self._session.delete(
                    url, data=payload, params=params, timeout=self.config.timeout,
                )
            case _:
                raise ValueError(f"Unsupported HTTP method: {method}")

    def _parse_response(self, resp: requests.Response, method: str) -> Any:
        """Inspect Content-Type and return the appropriate Python object."""
        content_type = resp.headers.get("content-type", "")

        if content_type.startswith("application/json") and resp.text:
            return resp.json()

        if content_type.startswith("image/"):
            return resp

        if method in ("delete", "post", "put"):
            return None

        # Non-JSON, non-image GET -- likely a stale-auth redirect or bad image.
        content_length = resp.headers.get("content-length", "")
        if content_length == "0" or resp.status_code == 404 or not resp.text.strip():
            logger.debug("Zero-byte or 404 or empty response; raising BAD_IMAGE")
            raise ValueError("BAD_IMAGE")

        logger.debug("Unexpected content-type %r status=%d body=%r; raising RELOGIN",
                     content_type, resp.status_code, resp.text[:500])
        raise ValueError("RELOGIN")

    def _handle_http_error(
        self,
        exc: requests.HTTPError,
        url: str,
        method: str,
        params: dict[str, str],
        payload: dict[str, Any] | None,
        reauth: bool,
    ) -> Any:
        """Handle an HTTP error response."""
        status = exc.response.status_code if exc.response is not None else 0
        logger.debug("HTTP error %d for %s", status, url)

        if status == 401 and reauth:
            logger.debug("Got 401; attempting relogin and retry")
            self._auth.handle_401()
            return self.request(url, method, params, payload, reauth=False)

        if status == 404:
            logger.debug("Got 404; raising BAD_IMAGE")
            raise ValueError("BAD_IMAGE")

        raise exc

    def _handle_value_error(
        self,
        exc: ValueError,
        url: str,
        method: str,
        params: dict[str, str],
        payload: dict[str, Any] | None,
        reauth: bool,
    ) -> Any:
        """Handle internal ValueError signals (RELOGIN / BAD_IMAGE)."""
        msg = str(exc)

        if msg == "RELOGIN" and reauth:
            logger.debug("RELOGIN signal; attempting relogin and retry")
            self._auth.handle_401()
            return self.request(url, method, params, payload, reauth=False)

        # BAD_IMAGE or exhausted reauth -- propagate
        raise exc
