"""JWT authentication for the pyzm serve API."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    from pyzm.models.config import ServerConfig

_bearer_scheme = HTTPBearer()


def create_login_route(config: "ServerConfig"):
    """Return a login endpoint handler bound to *config*.

    When ``config.auth_enabled`` is ``False`` the route accepts any
    credentials so that clients with ``ml_user`` / ``ml_password``
    configured don't fail with a 404.
    """

    def login(credentials: dict):
        username = credentials.get("username", "")
        password = credentials.get("password", "")
        if config.auth_enabled and (
            username != config.auth_username
            or password != config.auth_password.get_secret_value()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        now = time.time()
        expires = config.token_expiry_seconds
        payload = {"sub": username, "iat": now, "exp": now + expires}
        token = jwt.encode(payload, config.token_secret, algorithm="HS256")
        return {"access_token": token, "expires": expires}

    return login


def create_token_dependency(config: "ServerConfig"):
    """Return a FastAPI dependency that verifies Bearer tokens."""

    def verify_token(
        creds: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    ) -> str:
        try:
            payload = jwt.decode(
                creds.credentials, config.token_secret, algorithms=["HS256"]
            )
            return payload["sub"]
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

    return verify_token
