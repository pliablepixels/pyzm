"""Fixtures and skip logic for live ZoneMinder E2E tests.

These tests hit a real ZoneMinder server. Configuration is loaded from
environment variables or a `.env.zm_e2e` file in the repo root.

Run conventions:
    # readonly (default) -- skips automatically if no ZM_API_URL
    pytest tests/test_zm_e2e/ -v

    # include write-tier tests
    ZM_E2E_WRITE=1 pytest tests/test_zm_e2e/ -v

    # normal dev -- zm_e2e tests auto-skip when env file / ZM_API_URL is unset
    pytest tests/
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import pytest
import requests

from pyzm.client import ZMClient
from pyzm.models.config import ZMClientConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-file loading
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.zm_e2e"


def _load_env_file() -> dict[str, str]:
    """Parse .env.zm_e2e (KEY=VALUE lines). Ignores comments and blanks."""
    if not _ENV_FILE.is_file():
        return {}
    values: dict[str, str] = {}
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        if key:
            values[key.strip()] = val.strip()
    return values


# Load file values once; env vars always take precedence.
_file_values = _load_env_file()

# Propagate PYZM_CONFPATH to os.environ so that pyzm.zm.db picks it up
# at import time (it reads os.environ once when the module loads).
_conf_path = _file_values.get("PYZM_CONFPATH", "")
if _conf_path and "PYZM_CONFPATH" not in os.environ:
    os.environ["PYZM_CONFPATH"] = _conf_path


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, _file_values.get(name, default))


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

_ZM_API_URL = _get("ZM_API_URL")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip zm_e2e tests when the ZM server isn't configured."""
    skip_no_server = pytest.mark.skip(reason="ZM_API_URL not set (no .env.zm_e2e)")
    skip_no_write = pytest.mark.skip(reason="ZM_E2E_WRITE != 1 (write tests disabled)")
    write_enabled = _get("ZM_E2E_WRITE") == "1"

    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "zm_e2e" not in markers and "zm_e2e_write" not in markers:
            continue
        if not _ZM_API_URL:
            item.add_marker(skip_no_server)
        elif "zm_e2e_write" in markers and not write_enabled:
            item.add_marker(skip_no_write)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_zm(
    api_url: str,
    timeout: int = 90,
    verify_ssl: bool = False,
    user: str = "admin",
    password: str = "admin",
) -> bool:
    """Poll the ZM API via login until it responds or timeout expires."""
    login_url = f"{api_url}/host/login.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.post(
                login_url,
                data={"user": user, "pass": password},
                timeout=5,
                verify=verify_ssl,
            )
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def zm_db_available() -> tuple[bool, str]:
    """Check if we can connect to the ZM database via get_zm_db().

    Returns (ok, reason) — reason is empty on success, descriptive on failure.
    """
    try:
        from pyzm.zm.db import get_zm_db
        conn = get_zm_db()
        if conn is not None:
            conn.close()
            return True, ""
        return False, "get_zm_db() returned None (missing mysql-connector-python?)"
    except PermissionError as exc:
        return False, (f"Permission denied: {exc.filename or exc} — "
                       "sudo /opt/zoneminder/venv/bin/pip install pytest && "
                       "sudo -u www-data /opt/zoneminder/venv/bin/python -m pytest")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def zm_e2e_config() -> ZMClientConfig:
    """ZMClientConfig built from env vars / .env.zm_e2e."""
    if not _ZM_API_URL:
        pytest.skip("ZM_API_URL not set")
    return ZMClientConfig(
        api_url=_ZM_API_URL,
        user=_get("ZM_USER", "admin") or None,
        password=_get("ZM_PASSWORD", "admin") or None,
        verify_ssl=_get("ZM_VERIFY_SSL", "false").lower() in ("1", "true", "yes"),
    )


@pytest.fixture(scope="session")
def zm_client(zm_e2e_config: ZMClientConfig) -> ZMClient:
    """Session-scoped ZMClient for readonly tests. Logs in once."""
    client = ZMClient(config=zm_e2e_config)
    _record("Server", "API URL", zm_e2e_config.api_url)
    _record("Server", "ZM version", client.zm_version or "unknown")
    _record("Server", "API version", client.api_version or "unknown")
    return client


@pytest.fixture
def zm_client_fresh(zm_e2e_config: ZMClientConfig) -> ZMClient:
    """Fresh ZMClient per test — for testing login/auth behaviour."""
    return ZMClient(config=zm_e2e_config)


@pytest.fixture(scope="session")
def any_event(zm_client: ZMClient):
    """First available Event, skips if none exist."""
    events = zm_client.events(limit=1)
    if not events:
        pytest.skip("No events on ZM server")
    ev = events[0]
    _record("Test targets", "Event", f"id={ev.id}  monitor={ev.monitor_id}  "
            f"frames={ev.frames}  alarm_frames={ev.alarm_frames}")
    return ev


@pytest.fixture(scope="session")
def any_monitor(zm_client: ZMClient, any_event):
    """Monitor that owns any_event. Guarantees the monitor has at least one event."""
    mon = zm_client.monitor(any_event.monitor_id)
    _record("Test targets", "Monitor",
            f"id={mon.id}  name={mon.name!r}  {mon.width}x{mon.height}  "
            f"function={mon.function}")
    return mon


@pytest.fixture(scope="session")
def object_event(zm_client: ZMClient):
    """An event whose notes contain 'detected' (i.e. ZM flagged objects).
    Skips if no such events exist on the server."""
    events = zm_client.events(object_only=True, limit=1)
    if not events:
        pytest.skip("No object-detected events on ZM server")
    ev = events[0]
    notes_preview = (ev.notes or "")[:80]
    _record("Test targets", "Object event",
            f"id={ev.id}  notes={notes_preview!r}")
    return ev


@pytest.fixture
def note_restorer(any_event):
    """Save event notes before test, restore them in teardown."""
    original = any_event.notes
    yield any_event
    any_event.update_notes(original or "")


# ---------------------------------------------------------------------------
# E2E summary report
# ---------------------------------------------------------------------------

# Shared dict for tests to record interesting results into.
# Keys are section names, values are lists of (label, detail) tuples.
_summary: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()


@pytest.fixture(scope="session")
def e2e_summary() -> OrderedDict[str, list[tuple[str, str]]]:
    """Shared summary dict. Tests append (label, detail) tuples."""
    return _summary


def _record(section: str, label: str, detail: str) -> None:
    """Helper for fixtures to record summary lines."""
    _summary.setdefault(section, []).append((label, detail))


def pytest_terminal_summary(
    terminalreporter: "TerminalReporter",
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print a ZM E2E results summary after the test run."""
    # Collect skip reasons for zm_e2e tests
    skipped = terminalreporter.stats.get("skipped", [])
    skip_reasons: dict[str, list[str]] = {}
    for report in skipped:
        if "test_zm_e2e" not in str(report.fspath):
            continue
        # report.longrepr is a tuple: (file, lineno, reason)
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reason = reason.removeprefix("Skipped: ")
        skip_reasons.setdefault(reason, []).append(report.nodeid.split("::")[-1])

    if not _summary and not skip_reasons:
        return
    tw = terminalreporter._tw
    tw.sep("=", "ZM E2E Summary")
    for section, items in _summary.items():
        tw.line(f"\n  {section}:", bold=True)
        for label, detail in items:
            tw.line(f"    {label}: {detail}")
    if skip_reasons:
        tw.line(f"\n  Skipped ({sum(len(v) for v in skip_reasons.values())} tests):", bold=True)
        for reason, tests in skip_reasons.items():
            tw.line(f"    {reason}  ({len(tests)} tests)")
