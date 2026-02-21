"""Tests for pyzm.client -- ZMClient high-level API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import StreamConfig, ZMClientConfig
from pyzm.models.zm import Event, Monitor, PTZCapabilities, Zone


# ===================================================================
# Helpers
# ===================================================================

def _make_mock_api():
    """Create a mock ZMAPI."""
    api = MagicMock()
    api.api_version = "2.0.0"
    api.zm_version = "1.36.12"
    api.api_url = "https://zm.example.com/zm/api"
    api.portal_url = "https://zm.example.com/zm"
    return api


def _sample_monitor_api_data(
    mid=1, name="Front Door", function="Modect",
    controllable="0", control_id=None,
):
    mon = {
        "Id": str(mid),
        "Name": name,
        "Function": function,
        "Enabled": "1",
        "Width": "1920",
        "Height": "1080",
        "Type": "Ffmpeg",
        "Controllable": controllable,
    }
    if control_id is not None:
        mon["ControlId"] = str(control_id)
    return {
        "Monitor": mon,
        "Monitor_Status": {
            "Status": "Connected",
            "CaptureFPS": "15.0",
            "Capturing": "Capturing",
        },
    }


def _sample_control_data(**overrides):
    """Return a control profile API response dict."""
    defaults = {
        "CanMove": "1", "CanMoveCon": "1", "CanMoveRel": "0", "CanMoveAbs": "0",
        "CanZoom": "1", "CanZoomCon": "1", "CanZoomRel": "0", "CanZoomAbs": "0",
        "HasPresets": "0", "NumPresets": "0", "HasHomePreset": "0",
    }
    defaults.update(overrides)
    return {"control": {"Control": defaults}}


def _sample_event_api_data(eid=12345, name="Event 12345", monitor_id=1):
    return {
        "Event": {
            "Id": str(eid),
            "Name": name,
            "MonitorId": str(monitor_id),
            "Cause": "Motion",
            "Notes": "",
            "StartTime": "2024-03-15 10:30:00",
            "EndTime": "2024-03-15 10:31:30",
            "Length": "90.5",
            "Frames": "270",
            "AlarmFrames": "45",
            "MaxScore": "97",
            "MaxScoreFrameId": "135",
            "StoragePath": "/var/cache/zoneminder/events",
        }
    }


# ===================================================================
# TestZMClient - Creation
# ===================================================================

class TestZMClientCreation:
    """Tests for ZMClient construction."""

    @patch("pyzm.client.ZMAPI")
    def test_creation_with_url(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api", user="admin", password="secret")

        mock_zmapi_cls.assert_called_once()
        assert client.api is not None

    @patch("pyzm.client.ZMAPI")
    def test_creation_with_config(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        cfg = ZMClientConfig(api_url="https://zm.example.com/zm/api")
        client = ZMClient(config=cfg)

        mock_zmapi_cls.assert_called_once()

    @patch("pyzm.client.ZMAPI")
    def test_creation_strips_trailing_slash(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api/", user="admin", password="secret")

        assert client._config.api_url == "https://zm.example.com/zm/api"

    @patch("pyzm.client.ZMAPI")
    def test_creation_no_api_url_no_config_raises(self, mock_zmapi_cls):
        from pyzm.client import ZMClient
        with pytest.raises(ValueError, match="Either 'api_url' or 'config'"):
            ZMClient()

    @patch("pyzm.client.ZMAPI")
    def test_properties(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.zm_version == "1.36.12"
        assert client.api_version == "2.0.0"


# ===================================================================
# TestZMClient - Monitors
# ===================================================================

class TestZMClientMonitors:
    """Tests for monitors-related methods."""

    @patch("pyzm.client.ZMAPI")
    def test_monitors_fetches_and_caches(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(2, "Back Yard"),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        monitors = client.monitors()
        assert len(monitors) == 2
        assert monitors[0].name == "Front Door"
        assert monitors[1].name == "Back Yard"

        # Second call should use cache (API not called again)
        mock_api.get.reset_mock()
        monitors2 = client.monitors()
        mock_api.get.assert_not_called()
        assert monitors2 is monitors

    @patch("pyzm.client.ZMAPI")
    def test_monitors_have_client_reference(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(1, "Front Door")]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitors()[0]
        assert m._client is client

    @patch("pyzm.client.ZMAPI")
    def test_monitors_force_reload(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(1, "Front Door")]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.monitors()  # first fetch
        mock_api.get.reset_mock()

        # Force reload should call API again
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(3, "Side Gate"),
            ]
        }
        monitors = client.monitors(force_reload=True)
        mock_api.get.assert_called_once()
        assert len(monitors) == 2

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_id_from_cache(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(2, "Back Yard"),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(2)
        assert m.id == 2
        assert m.name == "Back Yard"
        assert m._client is client

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_id_not_found_fallback(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        # monitors() returns only monitor 1
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(1, "Front Door")]},
            {"monitor": _sample_monitor_api_data(99, "Unknown")},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(99)
        assert m.id == 99
        assert m._client is client

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_name(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(2, "Back Yard"),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor("Back Yard")
        assert m.id == 2
        assert m.name == "Back Yard"

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_name_case_insensitive(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor("front door")
        assert m.id == 1
        assert m.name == "Front Door"

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_name_not_found_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"monitors": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found"):
            client.monitor("Nonexistent")

    @patch("pyzm.client.ZMAPI")
    def test_monitor_not_found_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": []},
            {},  # empty response for direct lookup
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found"):
            client.monitor(999)


# ===================================================================
# TestMonitor - OOP methods
# ===================================================================

class TestMonitorOOP:
    """Tests for Monitor resource methods (zones, arm, disarm, etc.)."""

    @patch("pyzm.client.ZMAPI")
    def test_monitor_get_zones(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(1, "Front Door")]},
            {"zones": [
                {"Zone": {"Name": "driveway", "Coords": "0,0 100,0 100,100 0,100"}},
            ]},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(1)
        zones = m.get_zones()
        assert len(zones) == 1
        assert zones[0].name == "driveway"
        mock_api.get.assert_called_with("zones/forMonitor/1.json")

    @patch("pyzm.client.ZMAPI")
    def test_monitor_arm(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"monitors": [_sample_monitor_api_data(5)]}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        mock_api.get.reset_mock()
        m.arm()
        mock_api.get.assert_called_with("monitors/alarm/id:5/command:on.json")

    @patch("pyzm.client.ZMAPI")
    def test_monitor_disarm(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"monitors": [_sample_monitor_api_data(5)]}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        mock_api.get.reset_mock()
        m.disarm()
        mock_api.get.assert_called_with("monitors/alarm/id:5/command:off.json")

    @patch("pyzm.client.ZMAPI")
    def test_monitor_alarm_status(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"monitors": [_sample_monitor_api_data(5)]}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        mock_api.get.reset_mock()
        m.alarm_status()
        mock_api.get.assert_called_with("monitors/alarm/id:5/command:status.json")

    @patch("pyzm.client.ZMAPI")
    def test_monitor_update(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(3, "Front Door")]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(3)
        m.update(Function="Modect", Enabled="1")
        mock_api.put.assert_called_once_with(
            "monitors/3.json",
            data={"Monitor[Function]": "Modect", "Monitor[Enabled]": "1"},
        )

    @patch("pyzm.client.ZMAPI")
    def test_monitor_update_invalidates_cache(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(1, "Front Door")]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitors()[0]
        assert client._monitors is not None

        m.update(Name="New Name")
        assert client._monitors is None  # cache invalidated

    @patch("pyzm.client.ZMAPI")
    def test_monitor_daemon_status(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(3)]},
            {"status": True},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(3)
        result = m.daemon_status()
        assert result == {"status": True}
        mock_api.get.assert_called_with(
            "monitors/daemonStatus/id:3/daemon:zmc.json"
        )

    @patch("pyzm.client.ZMAPI")
    def test_monitor_daemon_status_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(3)]},
            None,
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(3)
        assert m.daemon_status() == {}

    def test_monitor_events_delegates(self):
        """Monitor.events() should delegate to _client.events(monitor_id=...)."""
        mock_client = MagicMock()
        mock_client.events.return_value = []
        m = Monitor(id=7, name="Test", _client=mock_client)

        m.events(until="6 hours ago")
        mock_client.events.assert_called_once_with(monitor_id=7, until="6 hours ago")

    def test_monitor_delete_events_delegates(self):
        """Monitor.delete_events() should delegate to _client.delete_events(monitor_id=...)."""
        mock_client = MagicMock()
        mock_client.delete_events.return_value = 3
        m = Monitor(id=7, name="Test", _client=mock_client)

        count = m.delete_events(before="1 day ago")
        assert count == 3
        mock_client.delete_events.assert_called_once_with(monitor_id=7, before="1 day ago")

    def test_monitor_events_without_client_raises(self):
        """Monitor.events() without a client should raise RuntimeError."""
        m = Monitor(id=1, name="Test")
        with pytest.raises(RuntimeError, match="not bound"):
            m.events()

    def test_monitor_without_client_raises(self):
        """Monitor not obtained via ZMClient should raise RuntimeError."""
        m = Monitor(id=1, name="Test")
        with pytest.raises(RuntimeError, match="not bound"):
            m.get_zones()

    @patch("pyzm.client.ZMAPI")
    def test_streaming_url_mjpeg_with_overlap(self, mock_zmapi_cls):
        """Portal /zm + ZMS /zm/cgi-bin/nph-zms → deduplicated URL."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(3)]},
            {"config": {"Config": {"Name": "ZM_PATH_ZMS", "Value": "/zm/cgi-bin/nph-zms"}}},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(3)
        url = m.streaming_url()
        assert url == "https://zm.example.com/zm/cgi-bin/nph-zms?mode=jpeg&monitor=3"

    @patch("pyzm.client.ZMAPI")
    def test_streaming_url_mjpeg_no_overlap(self, mock_zmapi_cls):
        """Portal with no path → no deduplication needed."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(1)]},
            {"config": {"Config": {"Name": "ZM_PATH_ZMS", "Value": "/cgi-bin/nph-zms"}}},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/api")

        m = client.monitor(1)
        url = m.streaming_url()
        assert url == "https://zm.example.com/cgi-bin/nph-zms?mode=jpeg&monitor=1"

    @patch("pyzm.client.ZMAPI")
    def test_streaming_url_mjpeg_extra_params(self, mock_zmapi_cls):
        """Extra kwargs are appended as query params."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(2)]},
            {"config": {"Config": {"Name": "ZM_PATH_ZMS", "Value": "/zm/cgi-bin/nph-zms"}}},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(2)
        url = m.streaming_url(maxfps=5, scale=50)
        assert "mode=jpeg" in url
        assert "monitor=2" in url
        assert "maxfps=5" in url
        assert "scale=50" in url

    @patch("pyzm.client.ZMAPI")
    def test_streaming_url_config_not_found_raises(self, mock_zmapi_cls):
        """Missing ZM_PATH_ZMS config raises ValueError."""
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(1)]},
            {},  # config not found
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(1)
        with pytest.raises(ValueError, match="not found"):
            m.streaming_url()

    def test_streaming_url_unknown_protocol_raises(self):
        """Unknown protocol raises ValueError."""
        mock_client = MagicMock()
        m = Monitor(id=1, name="Test", _client=mock_client)
        with pytest.raises(ValueError, match="Unknown streaming protocol"):
            m.streaming_url(protocol="webrtc")

    def test_streaming_url_without_client_raises(self):
        """Monitor not bound to client raises RuntimeError."""
        m = Monitor(id=1, name="Test")
        with pytest.raises(RuntimeError, match="not bound"):
            m.streaming_url()

    @patch("pyzm.client.ZMAPI")
    def test_snapshot_url_with_overlap(self, mock_zmapi_cls):
        """Snapshot URL deduplicates /zm overlap, uses mode=single."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(3)]},
            {"config": {"Config": {"Name": "ZM_PATH_ZMS", "Value": "/zm/cgi-bin/nph-zms"}}},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(3)
        url = m.snapshot_url()
        assert url == "https://zm.example.com/zm/cgi-bin/nph-zms?mode=single&monitor=3"

    @patch("pyzm.client.ZMAPI")
    def test_snapshot_url_extra_params(self, mock_zmapi_cls):
        """Extra kwargs are appended as query params."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(2)]},
            {"config": {"Config": {"Name": "ZM_PATH_ZMS", "Value": "/zm/cgi-bin/nph-zms"}}},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(2)
        url = m.snapshot_url(scale=50)
        assert "mode=single" in url
        assert "monitor=2" in url
        assert "scale=50" in url

    def test_snapshot_url_without_client_raises(self):
        """Monitor not bound to client raises RuntimeError."""
        m = Monitor(id=1, name="Test")
        with pytest.raises(RuntimeError, match="not bound"):
            m.snapshot_url()


# ===================================================================
# TestMonitor - PTZ
# ===================================================================

class TestMonitorPTZ:
    """Tests for Monitor PTZ control methods."""

    def test_monitor_controllable_parsed(self):
        """Controllable and ControlId are parsed from API data."""
        data = _sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)
        m = Monitor.from_api_dict(data)
        assert m.controllable is True
        assert m.control_id == 3

    def test_monitor_not_controllable_by_default(self):
        """Controllable defaults to False when not set."""
        data = _sample_monitor_api_data(1, "Fixed Cam")
        m = Monitor.from_api_dict(data)
        assert m.controllable is False
        assert m.control_id is None

    @patch("pyzm.client.ZMAPI")
    def test_ptz_capabilities(self, mock_zmapi_cls):
        """ptz_capabilities() returns correct PTZCapabilities."""
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            {"control": {"Control": {
                "CanMove": "1", "CanMoveCon": "1", "CanMoveRel": "0",
                "CanZoom": "1", "CanZoomCon": "1",
                "HasPresets": "1", "NumPresets": "20", "HasHomePreset": "1",
            }}},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        caps = m.ptz_capabilities()
        assert isinstance(caps, PTZCapabilities)
        assert caps.can_move is True
        assert caps.can_move_con is True
        assert caps.can_move_rel is False
        assert caps.can_zoom is True
        assert caps.has_presets is True
        assert caps.num_presets == 20
        assert caps.has_home_preset is True
        mock_api.get.assert_called_with("controls/3.json")

    @patch("pyzm.client.ZMAPI")
    def test_ptz_capabilities_not_found_raises(self, mock_zmapi_cls):
        """ptz_capabilities() raises ValueError when control profile missing."""
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=99)]},
            {},  # control not found
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        with pytest.raises(ValueError, match="not found"):
            m.ptz_capabilities()

    def test_ptz_capabilities_no_control_id_raises(self):
        """ptz_capabilities() on monitor without control_id raises ValueError."""
        mock_client = MagicMock()
        m = Monitor(id=1, name="Fixed", controllable=True, control_id=None, _client=mock_client)
        with pytest.raises(ValueError, match="no control profile"):
            m.ptz_capabilities()

    @patch("pyzm.client.ZMAPI")
    def test_ptz_command_up(self, mock_zmapi_cls):
        """ptz('up') auto-detects con mode and sends moveConUp to portal."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanMoveCon="1", CanMoveRel="0"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("up")

        mock_api.request.assert_called_once()
        call_args = mock_api.request.call_args
        assert call_args[0][0] == "https://zm.example.com/zm/index.php"
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert params["control"] == "moveConUp"
        assert params["id"] == "5"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_command_relative_mode(self, mock_zmapi_cls):
        """ptz('down', mode='rel') sends moveRelDown."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("down", mode="rel")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "moveRelDown"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_zoom_in(self, mock_zmapi_cls):
        """ptz('zoom_in') auto-detects con mode and sends zoomConTele."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanZoomCon="1", CanZoomRel="0"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("zoom_in")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "zoomConTele"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_stop(self, mock_zmapi_cls):
        """ptz('stop') sends moveStop."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("stop")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "moveStop"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_home(self, mock_zmapi_cls):
        """ptz('home') sends presetHome."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("home")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "presetHome"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_preset(self, mock_zmapi_cls):
        """ptz('preset', preset=3) sends presetGoto3."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("preset", preset=3)

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "presetGoto3"

    @patch("pyzm.client.time")
    @patch("pyzm.client.ZMAPI")
    def test_ptz_stop_after(self, mock_zmapi_cls, mock_time):
        """ptz('up', stop_after=2) auto-detects con then sends moveConUp + moveStop."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanMoveCon="1"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("up", stop_after=2.0)

        assert mock_api.request.call_count == 2
        # First call: moveConUp
        first_params = mock_api.request.call_args_list[0][1].get("params") or mock_api.request.call_args_list[0][0][1]
        assert first_params["control"] == "moveConUp"
        # Second call: moveStop
        second_params = mock_api.request.call_args_list[1][1].get("params") or mock_api.request.call_args_list[1][0][1]
        assert second_params["control"] == "moveStop"
        mock_time.sleep.assert_called_once_with(2.0)

    @patch("pyzm.client.time")
    @patch("pyzm.client.ZMAPI")
    def test_ptz_stop_after_not_on_stop_command(self, mock_zmapi_cls, mock_time):
        """ptz('stop', stop_after=2) should NOT send a second stop."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("stop", stop_after=2.0)

        assert mock_api.request.call_count == 1
        mock_time.sleep.assert_not_called()

    def test_ptz_non_controllable_raises(self):
        """ptz() on non-controllable monitor raises ValueError."""
        mock_client = MagicMock()
        m = Monitor(id=1, name="Fixed", controllable=False, _client=mock_client)
        with pytest.raises(ValueError, match="not controllable"):
            m.ptz("up")

    def test_ptz_unbound_raises(self):
        """ptz() on unbound monitor raises RuntimeError."""
        m = Monitor(id=1, name="Test")
        with pytest.raises(RuntimeError, match="not bound"):
            m.ptz("up")

    def test_ptz_capabilities_unbound_raises(self):
        """ptz_capabilities() on unbound monitor raises RuntimeError."""
        m = Monitor(id=1, name="Test")
        with pytest.raises(RuntimeError, match="not bound"):
            m.ptz_capabilities()

    @patch("pyzm.client.ZMAPI")
    def test_ptz_auto_selects_rel_when_con_unsupported(self, mock_zmapi_cls):
        """ptz('left') auto-selects rel mode when con is not supported."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanMoveCon="0", CanMoveRel="1"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("left")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "moveRelLeft"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_auto_selects_abs_as_last_resort(self, mock_zmapi_cls):
        """ptz('up') auto-selects abs mode when con and rel are unsupported."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanMoveCon="0", CanMoveRel="0", CanMoveAbs="1"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("up")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "moveAbsUp"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_auto_zoom_selects_rel(self, mock_zmapi_cls):
        """ptz('zoom_in') auto-selects rel when zoom con unsupported."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanZoomCon="0", CanZoomRel="1"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("zoom_in")

        params = mock_api.request.call_args[1].get("params") or mock_api.request.call_args[0][1]
        assert params["control"] == "zoomRelTele"

    @patch("pyzm.client.ZMAPI")
    def test_ptz_auto_caches_capabilities(self, mock_zmapi_cls):
        """Auto-mode fetches capabilities once and caches for subsequent calls."""
        mock_api = _make_mock_api()
        mock_api.portal_url = "https://zm.example.com/zm"
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(5, "PTZ Cam", controllable="1", control_id=3)]},
            _sample_control_data(CanMoveCon="0", CanMoveRel="1"),
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        m = client.monitor(5)
        m.ptz("left")
        m.ptz("right")

        # get called twice: once for monitor, once for control (not again for second ptz)
        assert mock_api.get.call_count == 2
        assert mock_api.request.call_count == 2

    def test_ptz_unknown_command_raises(self):
        """ptz() with unknown command raises ValueError."""
        from pyzm.client import _ptz_command_name
        with pytest.raises(ValueError, match="Unknown PTZ command"):
            _ptz_command_name("fly")

    def test_ptz_unknown_mode_raises(self):
        """ptz() with unknown mode raises ValueError."""
        from pyzm.client import _ptz_command_name
        with pytest.raises(ValueError, match="Unknown PTZ mode"):
            _ptz_command_name("up", mode="turbo")


# ===================================================================
# TestZMClient - Events
# ===================================================================

class TestZMClientEvents:
    """Tests for events-related methods."""

    @patch("pyzm.client.ZMAPI")
    def test_events_basic(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "events": [
                _sample_event_api_data(100),
                _sample_event_api_data(101),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        events = client.events()
        assert len(events) == 2
        assert events[0].id == 100
        assert events[1].id == 101

    @patch("pyzm.client.ZMAPI")
    def test_events_have_client_reference(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "events": [_sample_event_api_data(100)]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.events()[0]
        assert ev._client is client

    @patch("pyzm.client.ZMAPI")
    def test_events_filter_by_monitor(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"events": [_sample_event_api_data(100)]}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.events(monitor_id=2)

        # Verify the URL uses CakePHP path filter syntax
        call_url = mock_api.get.call_args[0][0]
        assert "MonitorId:2" in call_url

    @patch("pyzm.client.ZMAPI")
    def test_events_filter_building(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"events": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.events(
            monitor_id=1,
            min_alarm_frames=5,
            object_only=True,
            limit=50,
        )

        call_url = mock_api.get.call_args[0][0]
        assert "MonitorId:1" in call_url
        assert "AlarmFrames >=:5" in call_url
        assert "Notes REGEXP:detected" in call_url
        call_params = mock_api.get.call_args[1].get("params", {})
        assert call_params["limit"] == "50"

    @patch("pyzm.client.ZMAPI")
    def test_event_single_fetch(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "event": _sample_event_api_data(42)
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(42)
        assert ev.id == 42
        assert ev._client is client

    @patch("pyzm.client.ZMAPI")
    def test_event_not_found_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found"):
            client.event(99999)


# ===================================================================
# TestEvent - OOP methods
# ===================================================================

class TestEventOOP:
    """Tests for Event resource methods (frames, update_notes, tag, etc.)."""

    @patch("pyzm.client.ZMAPI")
    def test_event_get_frames(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"event": _sample_event_api_data(100)},
            {"frames": [
                {"Frame": {"FrameId": "1", "EventId": "100", "Type": "Normal", "Score": "0", "Delta": "0.5"}},
                {"Frame": {"FrameId": "5", "EventId": "100", "Type": "Alarm", "Score": "85", "Delta": "2.1"}},
                {"Frame": {"FrameId": "10", "EventId": "100", "Type": "Normal", "Score": "0", "Delta": "4.0"}},
            ]},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(100)
        frames = ev.get_frames()
        assert len(frames) == 3
        assert frames[0].frame_id == 1
        assert frames[0].type == "Normal"
        assert frames[1].frame_id == 5
        assert frames[1].type == "Alarm"
        assert frames[1].score == 85
        assert frames[1].delta == 2.1

    @patch("pyzm.client.ZMAPI")
    def test_event_get_frames_calls_correct_endpoint(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"event": _sample_event_api_data(12345)},
            {"frames": []},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(12345)
        ev.get_frames()
        mock_api.get.assert_called_with("frames/index/EventId:12345.json")

    @patch("pyzm.client.ZMAPI")
    def test_event_get_frames_empty_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"event": _sample_event_api_data(999)},
            {"frames": []},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(999)
        assert ev.get_frames() == []

    @patch("pyzm.client.ZMAPI")
    def test_event_get_frames_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"event": _sample_event_api_data(999)},
            None,
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(999)
        assert ev.get_frames() == []

    @patch("pyzm.client.ZMAPI")
    def test_event_get_frames_flat_dict(self, mock_zmapi_cls):
        """Frame data without wrapping 'Frame' key."""
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"event": _sample_event_api_data(200)},
            {"frames": [
                {"FrameId": "42", "EventId": "200", "Type": "Alarm", "Score": "95", "Delta": "1.5"},
            ]},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(200)
        frames = ev.get_frames()
        assert len(frames) == 1
        assert frames[0].frame_id == 42
        assert frames[0].score == 95

    @patch("pyzm.client.ZMAPI")
    def test_event_update_notes(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(12345)
        ev.update_notes("person:97% detected")

        mock_api.put.assert_called_once_with(
            "events/12345.json",
            data={"Event[Notes]": "person:97% detected"},
        )

    @patch("pyzm.client.FrameExtractor")
    @patch("pyzm.client.ZMAPI")
    def test_event_extract_frames(self, mock_zmapi_cls, mock_extractor_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.models.zm import Frame as ZMFrame
        mock_frame = ZMFrame(frame_id=1, event_id=12345)
        mock_image = MagicMock()
        mock_image.shape = (480, 640, 3)

        mock_extractor = MagicMock()
        mock_extractor.extract_frames.return_value = [(mock_frame, mock_image)]
        mock_extractor.original_shape = (480, 640)
        mock_extractor_cls.return_value = mock_extractor

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(12345)
        frames, image_dims = ev.extract_frames()
        assert len(frames) == 1
        assert frames[0][0] == 1  # frame_id
        assert frames[0][1] is mock_image
        assert image_dims["original"] == (480, 640)
        assert image_dims["resized"] is None  # no resize happened

    @patch("pyzm.client.FrameExtractor")
    @patch("pyzm.client.ZMAPI")
    def test_event_extract_frames_with_config(self, mock_zmapi_cls, mock_extractor_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        mock_extractor = MagicMock()
        mock_extractor.extract_frames.return_value = []
        mock_extractor_cls.return_value = mock_extractor

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(12345)
        sc = StreamConfig(frame_set=["1", "5"], max_frames=2)
        ev.extract_frames(stream_config=sc)

        # Verify FrameExtractor was created with the correct config
        mock_extractor_cls.assert_called_once()
        call_kwargs = mock_extractor_cls.call_args
        assert call_kwargs[1]["stream_config"] is sc or call_kwargs[0][1] is sc

    @patch("pyzm.client.ZMAPI")
    def test_event_delete(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(42)}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        ev = client.event(42)
        ev.delete()
        mock_api.delete.assert_called_once_with("events/42.json")

    def test_event_without_client_raises(self):
        """Event not obtained via ZMClient should raise RuntimeError."""
        ev = Event(id=1)
        with pytest.raises(RuntimeError, match="not bound"):
            ev.get_frames()


# ===================================================================
# TestZMClient - tag_event (via Event.tag)
# ===================================================================

class TestEventTag:
    """Tests for Event.tag() (direct DB implementation)."""

    @patch("pyzm.client.ZMAPI")
    def test_tag_creates_new_tag(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # tag doesn't exist
        mock_cursor.lastrowid = 7

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")
        client._get_db = MagicMock(return_value=mock_conn)
        ev = client.event(12345)
        ev.tag(["person"])

        # Should INSERT new tag then link it
        calls = mock_cursor.execute.call_args_list
        assert any("SELECT Id FROM Tags" in str(c) for c in calls)
        assert any("INSERT INTO Tags" in str(c) for c in calls)
        assert any("INSERT INTO Events_Tags" in str(c) for c in calls)
        mock_conn.commit.assert_called()

    @patch("pyzm.client.ZMAPI")
    def test_tag_links_existing_tag(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"Id": 42}  # tag exists

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")
        client._get_db = MagicMock(return_value=mock_conn)
        ev = client.event(12345)
        ev.tag(["person"])

        calls = mock_cursor.execute.call_args_list
        # Should UPDATE existing tag's LastAssignedDate, then link
        assert any("UPDATE Tags SET LastAssignedDate" in str(c) for c in calls)
        assert any("INSERT INTO Events_Tags" in str(c) for c in calls)
        # Should NOT insert a new tag
        assert not any("INSERT INTO Tags" in str(c) and "Name" in str(c) for c in calls)
        mock_conn.commit.assert_called()

    @patch("pyzm.client.ZMAPI")
    def test_tag_deduplicates_labels(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_cursor.lastrowid = 1

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")
        client._get_db = MagicMock(return_value=mock_conn)
        ev = client.event(12345)
        ev.tag(["person", "person", "car", "car"])

        # Should only process 2 unique labels, not 4
        select_calls = [c for c in mock_cursor.execute.call_args_list
                        if "SELECT Id FROM Tags" in str(c)]
        assert len(select_calls) == 2

    @patch("pyzm.client.ZMAPI")
    def test_tag_empty_labels(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")
        ev = client.event(12345)
        ev.tag([])
        # No DB calls should happen

    @patch("pyzm.client.ZMAPI")
    def test_tag_no_db_graceful(self, mock_zmapi_cls):
        """When DB is unavailable, tag() logs warning and returns."""
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"event": _sample_event_api_data(12345)}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")
        client._get_db = MagicMock(return_value=None)
        ev = client.event(12345)
        ev.tag(["person"])  # should not raise


# ===================================================================
# TestZMClient - delete_events (bulk)
# ===================================================================

class TestZMClientDeleteEvents:
    """Tests for delete_events (bulk query+delete)."""

    @patch("pyzm.client.ZMAPI")
    def test_delete_events_queries_then_deletes(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "events": [
                _sample_event_api_data(100),
                _sample_event_api_data(101),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        count = client.delete_events(monitor_id=1, limit=50)
        assert count == 2
        assert mock_api.delete.call_count == 2
        mock_api.delete.assert_any_call("events/100.json")
        mock_api.delete.assert_any_call("events/101.json")

    @patch("pyzm.client.ZMAPI")
    def test_delete_events_returns_zero_when_none_match(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"events": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        count = client.delete_events(before="2020-01-01")
        assert count == 0
        mock_api.delete.assert_not_called()


# ===================================================================
# TestZMClient - State Management
# ===================================================================

class TestZMClientState:
    """Tests for state management."""

    @patch("pyzm.client.ZMAPI")
    def test_set_state(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.set_state("restart")
        mock_api.get.assert_called_with("states/change/restart.json")

    @patch("pyzm.client.ZMAPI")
    def test_start(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.start()
        mock_api.get.assert_called_with("states/change/start.json")

    @patch("pyzm.client.ZMAPI")
    def test_stop(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.stop()
        mock_api.get.assert_called_with("states/change/stop.json")

    @patch("pyzm.client.ZMAPI")
    def test_restart(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.restart()
        mock_api.get.assert_called_with("states/change/restart.json")


# ===================================================================
# TestZMClient - System Health
# ===================================================================

class TestZMClientSystemHealth:
    """Tests for is_running, system_load, disk_usage, timezone."""

    @patch("pyzm.client.ZMAPI")
    def test_is_running_true(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"result": 1}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.is_running() is True
        mock_api.get.assert_called_with("host/daemonCheck.json")

    @patch("pyzm.client.ZMAPI")
    def test_is_running_false(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"result": 0}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.is_running() is False

    @patch("pyzm.client.ZMAPI")
    def test_is_running_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = None
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.is_running() is False

    @patch("pyzm.client.ZMAPI")
    def test_system_load(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"load": [0.5, 1.2, 0.8]}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        load = client.system_load()
        assert load == {"1min": 0.5, "5min": 1.2, "15min": 0.8}
        mock_api.get.assert_called_with("host/getLoad.json")

    @patch("pyzm.client.ZMAPI")
    def test_system_load_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = None
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.system_load() == {}

    @patch("pyzm.client.ZMAPI")
    def test_disk_usage(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"usage": {"/": 45.2}}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        result = client.disk_usage()
        assert result == {"usage": {"/": 45.2}}
        mock_api.get.assert_called_with("host/getDiskPercent.json")

    @patch("pyzm.client.ZMAPI")
    def test_disk_usage_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = None
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.disk_usage() == {}

    @patch("pyzm.client.ZMAPI")
    def test_timezone_tz_key(self, mock_zmapi_cls):
        """ZM 1.38+ uses 'tz' key."""
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"tz": "America/New_York"}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.timezone() == "America/New_York"
        mock_api.get.assert_called_with("host/getTimeZone.json")

    @patch("pyzm.client.ZMAPI")
    def test_timezone_legacy_key(self, mock_zmapi_cls):
        """Older ZM versions use 'timezone' key."""
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"timezone": "US/Eastern"}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.timezone() == "US/Eastern"

    @patch("pyzm.client.ZMAPI")
    def test_timezone_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = None
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.timezone() == ""


# ===================================================================
# TestZMClient - Configs
# ===================================================================

class TestZMClientConfigs:
    """Tests for configs, config, set_config."""

    @patch("pyzm.client.ZMAPI")
    def test_configs_list(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "configs": [
                {"Config": {"Id": "1", "Name": "ZM_LANG_DEFAULT", "Value": "en_us"}},
                {"Config": {"Id": "2", "Name": "ZM_OPT_USE_AUTH", "Value": "1"}},
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        cfgs = client.configs()
        assert len(cfgs) == 2
        assert cfgs[0]["Name"] == "ZM_LANG_DEFAULT"
        assert cfgs[1]["Name"] == "ZM_OPT_USE_AUTH"
        mock_api.get.assert_called_with("configs.json")

    @patch("pyzm.client.ZMAPI")
    def test_configs_empty(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"configs": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.configs() == []

    @patch("pyzm.client.ZMAPI")
    def test_config_by_name_wrapped(self, mock_zmapi_cls):
        """Older ZM returns Config wrapper with Name/Id."""
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "config": {"Config": {"Id": "5", "Name": "ZM_TIMEZONE", "Value": "US/Eastern"}}
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        cfg = client.config("ZM_TIMEZONE")
        assert cfg["Name"] == "ZM_TIMEZONE"
        assert cfg["Value"] == "US/Eastern"
        mock_api.get.assert_called_with("configs/viewByName/ZM_TIMEZONE.json")

    @patch("pyzm.client.ZMAPI")
    def test_config_by_name_flat(self, mock_zmapi_cls):
        """ZM 1.38+ returns flat config without Name key."""
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "config": {"Value": "en_gb"}
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        cfg = client.config("ZM_LANG_DEFAULT")
        assert cfg["Name"] == "ZM_LANG_DEFAULT"  # injected by setdefault
        assert cfg["Value"] == "en_gb"

    @patch("pyzm.client.ZMAPI")
    def test_config_not_found_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found"):
            client.config("ZM_NONEXISTENT")

    @patch("pyzm.client.ZMAPI")
    def test_set_config(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        # First call: config() to get the Id
        mock_api.get.return_value = {
            "config": {"Config": {"Id": "10", "Name": "ZM_TIMEZONE", "Value": "US/Eastern"}}
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        client.set_config("ZM_TIMEZONE", "US/Pacific")
        mock_api.put.assert_called_once_with(
            "configs/10.json",
            data={"Config[Value]": "US/Pacific"},
        )

    @patch("pyzm.client.ZMAPI")
    def test_set_config_no_id_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        # Config exists but has no Id field
        mock_api.get.return_value = {
            "config": {"Config": {"Name": "ZM_BROKEN"}}
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found or has no Id"):
            client.set_config("ZM_BROKEN", "value")


# ===================================================================
# TestZMClient - States listing
# ===================================================================

class TestZMClientStates:
    """Tests for states()."""

    @patch("pyzm.client.ZMAPI")
    def test_states_list(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "states": [
                {"State": {"Id": "1", "Name": "default", "Definition": ""}},
                {"State": {"Id": "2", "Name": "home", "Definition": "1:Modect"}},
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        result = client.states()
        assert len(result) == 2
        assert result[0]["Name"] == "default"
        assert result[1]["Name"] == "home"
        mock_api.get.assert_called_with("states.json")

    @patch("pyzm.client.ZMAPI")
    def test_states_empty(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"states": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.states() == []

    @patch("pyzm.client.ZMAPI")
    def test_states_none_response(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = None
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.states() == []


# ===================================================================
# TestZMClient - Servers & Storage
# ===================================================================

class TestZMClientServersStorage:
    """Tests for servers() and storage()."""

    @patch("pyzm.client.ZMAPI")
    def test_servers_list(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "servers": [
                {"Server": {"Id": "1", "Name": "zm-primary", "Hostname": "zm1.local"}},
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        result = client.servers()
        assert len(result) == 1
        assert result[0]["Name"] == "zm-primary"
        mock_api.get.assert_called_with("servers.json")

    @patch("pyzm.client.ZMAPI")
    def test_servers_empty(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"servers": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.servers() == []

    @patch("pyzm.client.ZMAPI")
    def test_storage_list(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "storage": [
                {"Storage": {"Id": "1", "Name": "Default", "Path": "/var/cache/zoneminder/events"}},
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        result = client.storage()
        assert len(result) == 1
        assert result[0]["Name"] == "Default"
        mock_api.get.assert_called_with("storage.json")

    @patch("pyzm.client.ZMAPI")
    def test_storage_empty(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"storage": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(api_url="https://zm.example.com/zm/api")

        assert client.storage() == []
