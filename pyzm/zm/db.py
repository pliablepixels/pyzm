"""Direct MySQL connection to the ZoneMinder database.

Reads credentials from ``/etc/zm/zm.conf`` (and ``conf.d/*.conf``) —
the same files that ZM itself uses.  Explicit overrides take precedence.
"""

from __future__ import annotations

import configparser
import glob
import logging
import os

logger = logging.getLogger("pyzm.zm")

_CONF_PATH = os.environ.get("PYZM_CONFPATH", "/etc/zm")


def _read_zm_conf(conf_path: str = _CONF_PATH) -> dict[str, str]:
    """Parse ZM config files and return DB credentials."""
    files = sorted(glob.glob(os.path.join(conf_path, "conf.d", "*.conf")))
    files.insert(0, os.path.join(conf_path, "zm.conf"))

    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#",)
    )
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            parser.read_string("[zm_root]\n" + fh.read())

    section = parser["zm_root"] if parser.has_section("zm_root") else {}
    return {
        "user": section.get("ZM_DB_USER", "zmuser"),
        "password": section.get("ZM_DB_PASS", "zmpass"),
        "host": section.get("ZM_DB_HOST", "localhost"),
        "database": section.get("ZM_DB_NAME", "zm"),
    }


def get_zm_db(
    db_user: str | None = None,
    db_password: str | None = None,
    db_host: str | None = None,
    db_name: str | None = None,
    conf_path: str | None = None,
):
    """Return a ``mysql.connector`` connection to the ZM database, or ``None``.

    Merge strategy: always try to read zm.conf first. If it's unreadable
    (permission denied, missing), log a warning and continue. Then overlay
    any explicit values — explicit values win over zm.conf values. Final
    fallback for host is ``"localhost"``, for db name is ``"zm"``.
    """
    try:
        import mysql.connector
    except ImportError:
        logger.warning("mysql-connector-python not installed, DB access unavailable")
        return None

    # Start with zm.conf values (best-effort)
    try:
        creds = _read_zm_conf(conf_path or _CONF_PATH)
    except (PermissionError, OSError) as exc:
        logger.warning("Could not read zm.conf: %s", exc)
        creds = {"user": "zmuser", "password": "zmpass", "host": "localhost", "database": "zm"}

    # Overlay explicit overrides
    if db_user is not None:
        creds["user"] = db_user
    if db_password is not None:
        creds["password"] = db_password
    if db_host is not None:
        creds["host"] = db_host
    if db_name is not None:
        creds["database"] = db_name

    host = creds["host"]
    port = 3306

    # ZM_DB_HOST can be "hostname:port" or "hostname:/path/to/socket"
    if ":" in host:
        host, suffix = host.split(":", 1)
        if suffix.startswith("/"):
            # Unix socket path — pass as unix_socket
            try:
                return mysql.connector.connect(
                    user=creds["user"],
                    password=creds["password"],
                    database=creds["database"],
                    unix_socket=suffix,
                )
            except mysql.connector.Error as exc:
                logger.warning("DB connect via socket %s failed: %s", suffix, exc)
                return None
        else:
            port = int(suffix)

    try:
        return mysql.connector.connect(
            host=host,
            port=port,
            user=creds["user"],
            password=creds["password"],
            database=creds["database"],
        )
    except mysql.connector.Error as exc:
        logger.warning("DB connect to %s:%s failed: %s", host, port, exc)
        return None
