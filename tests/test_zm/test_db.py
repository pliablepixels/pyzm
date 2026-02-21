"""Tests for pyzm.zm.db -- ZM database connection with credential overrides."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.zm.db import _read_zm_conf, get_zm_db


class TestReadZmConf:
    """Tests for _read_zm_conf() parsing."""

    def test_parses_zm_conf(self, tmp_path):
        conf = tmp_path / "zm.conf"
        conf.write_text("ZM_DB_USER=myuser\nZM_DB_PASS=mypass\nZM_DB_HOST=dbhost\nZM_DB_NAME=mydb\n")
        (tmp_path / "conf.d").mkdir()

        creds = _read_zm_conf(str(tmp_path))
        assert creds["user"] == "myuser"
        assert creds["password"] == "mypass"
        assert creds["host"] == "dbhost"
        assert creds["database"] == "mydb"

    def test_defaults_when_conf_missing(self, tmp_path):
        creds = _read_zm_conf(str(tmp_path / "nonexistent"))
        assert creds["user"] == "zmuser"
        assert creds["password"] == "zmpass"
        assert creds["host"] == "localhost"
        assert creds["database"] == "zm"

    def test_conf_d_overrides_main(self, tmp_path):
        conf = tmp_path / "zm.conf"
        conf.write_text("ZM_DB_USER=base\n")
        confd = tmp_path / "conf.d"
        confd.mkdir()
        (confd / "01-override.conf").write_text("ZM_DB_USER=overridden\n")

        creds = _read_zm_conf(str(tmp_path))
        assert creds["user"] == "overridden"


class TestGetZmDb:
    """Tests for get_zm_db() with explicit credential overrides."""

    @patch("pyzm.zm.db._read_zm_conf")
    @patch("mysql.connector.connect")
    def test_explicit_overrides_win(self, mock_connect, mock_conf):
        """Explicit params override zm.conf values."""
        mock_conf.return_value = {
            "user": "conf_user", "password": "conf_pass",
            "host": "conf_host", "database": "conf_db",
        }

        get_zm_db(db_user="my_user", db_password="my_pass", db_host="my_host", db_name="my_db")

        mock_connect.assert_called_once_with(
            host="my_host", port=3306,
            user="my_user", password="my_pass", database="my_db",
        )

    @patch("pyzm.zm.db._read_zm_conf")
    @patch("mysql.connector.connect")
    def test_partial_overrides(self, mock_connect, mock_conf):
        """Only specified overrides replace zm.conf values."""
        mock_conf.return_value = {
            "user": "conf_user", "password": "conf_pass",
            "host": "conf_host", "database": "conf_db",
        }

        get_zm_db(db_user="custom_user")

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["user"] == "custom_user"
        assert call_kwargs["password"] == "conf_pass"
        assert call_kwargs["host"] == "conf_host"
        assert call_kwargs["database"] == "conf_db"

    @patch("pyzm.zm.db._read_zm_conf")
    @patch("mysql.connector.connect")
    def test_no_overrides_uses_conf(self, mock_connect, mock_conf):
        """No overrides = pure zm.conf values."""
        mock_conf.return_value = {
            "user": "zmuser", "password": "zmpass",
            "host": "localhost", "database": "zm",
        }

        get_zm_db()

        mock_connect.assert_called_once_with(
            host="localhost", port=3306,
            user="zmuser", password="zmpass", database="zm",
        )

    @patch("mysql.connector.connect")
    def test_conf_path_override(self, mock_connect, tmp_path):
        """conf_path redirects zm.conf reading to a custom directory."""
        conf = tmp_path / "zm.conf"
        conf.write_text("ZM_DB_USER=custom_dir_user\nZM_DB_PASS=p\nZM_DB_HOST=h\nZM_DB_NAME=d\n")
        (tmp_path / "conf.d").mkdir()

        get_zm_db(conf_path=str(tmp_path))

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["user"] == "custom_dir_user"

    @patch("pyzm.zm.db._read_zm_conf", side_effect=PermissionError("denied"))
    @patch("mysql.connector.connect")
    def test_permission_denied_falls_back(self, mock_connect, mock_conf):
        """When zm.conf is unreadable, defaults are used then overrides applied."""
        get_zm_db(db_user="override_user")

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["user"] == "override_user"
        # Other fields should use the built-in fallbacks
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["database"] == "zm"

    @patch("pyzm.zm.db._read_zm_conf", side_effect=PermissionError("denied"))
    @patch("mysql.connector.connect")
    def test_permission_denied_debug_when_explicit(self, mock_connect, mock_conf, caplog):
        """zm.conf failure is debug-level (not warning) when explicit creds provided."""
        import logging
        with caplog.at_level(logging.DEBUG, logger="pyzm.zm"):
            get_zm_db(db_user="u", db_password="p")
        assert any(r.levelno == logging.DEBUG and "explicit credentials" in r.message for r in caplog.records)
        assert not any(r.levelno == logging.WARNING and "zm.conf" in r.message for r in caplog.records)

    @patch("pyzm.zm.db._read_zm_conf", side_effect=PermissionError("denied"))
    @patch("mysql.connector.connect")
    def test_permission_denied_warning_when_no_explicit(self, mock_connect, mock_conf, caplog):
        """zm.conf failure is warning-level when no explicit creds provided."""
        import logging
        with caplog.at_level(logging.DEBUG, logger="pyzm.zm"):
            get_zm_db()
        assert any(r.levelno == logging.WARNING and "zm.conf" in r.message for r in caplog.records)

    @patch("pyzm.zm.db._read_zm_conf")
    @patch("mysql.connector.connect")
    def test_host_with_port(self, mock_connect, mock_conf):
        """host:port syntax is parsed correctly."""
        mock_conf.return_value = {
            "user": "u", "password": "p",
            "host": "dbhost:3307", "database": "zm",
        }

        get_zm_db()

        mock_connect.assert_called_once_with(
            host="dbhost", port=3307,
            user="u", password="p", database="zm",
        )

    @patch("pyzm.zm.db._read_zm_conf")
    @patch("mysql.connector.connect")
    def test_host_with_socket(self, mock_connect, mock_conf):
        """host:/path/to/socket syntax is parsed correctly."""
        mock_conf.return_value = {
            "user": "u", "password": "p",
            "host": "localhost:/var/run/mysqld/mysqld.sock", "database": "zm",
        }

        get_zm_db()

        mock_connect.assert_called_once_with(
            user="u", password="p", database="zm",
            unix_socket="/var/run/mysqld/mysqld.sock",
        )


class TestClientGetDb:
    """Tests for ZMClient._get_db() helper."""

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_get_db_passes_config_creds(self, mock_get_zm_db, mock_zmapi_cls):
        mock_zmapi_cls.return_value = MagicMock()
        mock_get_zm_db.return_value = MagicMock()

        from pyzm.client import ZMClient
        client = ZMClient(
            api_url="https://zm.example.com/zm/api",
            db_user="myuser",
            db_password="mypass",
            db_host="myhost",
            db_name="mydb",
            conf_path="/custom/path",
        )

        client._get_db()

        mock_get_zm_db.assert_called_once_with(
            db_user="myuser",
            db_password="mypass",
            db_host="myhost",
            db_name="mydb",
            conf_path="/custom/path",
        )

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_get_db_passes_none_when_no_overrides(self, mock_get_zm_db, mock_zmapi_cls):
        mock_zmapi_cls.return_value = MagicMock()
        mock_get_zm_db.return_value = MagicMock()

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client._get_db()

        mock_get_zm_db.assert_called_once_with(
            db_user=None,
            db_password=None,
            db_host=None,
            db_name=None,
            conf_path=None,
        )

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_get_db_from_config_object(self, mock_get_zm_db, mock_zmapi_cls):
        """DB creds from ZMClientConfig are passed through."""
        mock_zmapi_cls.return_value = MagicMock()
        mock_get_zm_db.return_value = MagicMock()

        from pydantic import SecretStr
        from pyzm.models.config import ZMClientConfig
        from pyzm.client import ZMClient

        cfg = ZMClientConfig(
            api_url="https://zm.example.com/zm/api",
            db_user="cfguser",
            db_password=SecretStr("cfgpass"),
            db_host="cfghost",
        )
        client = ZMClient(config=cfg)
        client._get_db()

        mock_get_zm_db.assert_called_once_with(
            db_user="cfguser",
            db_password="cfgpass",
            db_host="cfghost",
            db_name=None,
            conf_path=None,
        )
