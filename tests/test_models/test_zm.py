"""Tests for pyzm.models.zm -- ZoneMinder data models."""

from __future__ import annotations

from datetime import datetime

import pytest

from pyzm.models.zm import Event, Frame, Monitor, MonitorStatus, PTZCapabilities, Zone


# ===================================================================
# TestZone
# ===================================================================

class TestZone:
    """Tests for the Zone dataclass."""

    def test_creation(self):
        z = Zone(name="driveway", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
        assert z.name == "driveway"
        assert len(z.points) == 4
        assert z.pattern is None

    def test_creation_with_pattern(self):
        z = Zone(
            name="front_door",
            points=[(10, 10), (200, 10), (200, 300), (10, 300)],
            pattern="(person|package)",
        )
        assert z.pattern == "(person|package)"

    def test_as_dict(self):
        z = Zone(
            name="garden",
            points=[(0, 0), (50, 0), (50, 50), (0, 50)],
            pattern="(dog|cat)",
        )
        d = z.as_dict()
        assert d["name"] == "garden"
        assert d["value"] == [(0, 0), (50, 0), (50, 50), (0, 50)]
        assert d["pattern"] == "(dog|cat)"

    def test_as_dict_no_pattern(self):
        z = Zone(name="full", points=[(0, 0), (640, 0), (640, 480), (0, 480)])
        d = z.as_dict()
        assert d["pattern"] is None

    def test_empty_points(self):
        z = Zone(name="empty", points=[])
        assert z.points == []


# ===================================================================
# TestFrame
# ===================================================================

class TestFrame:
    """Tests for the Frame dataclass."""

    def test_creation_defaults(self):
        f = Frame(frame_id=1, event_id=12345)
        assert f.frame_id == 1
        assert f.event_id == 12345
        assert f.type == ""
        assert f.score == 0
        assert f.delta == 0.0

    def test_creation_with_all_fields(self):
        f = Frame(
            frame_id=42,
            event_id=99999,
            type="Alarm",
            score=85,
            delta=1.25,
        )
        assert f.frame_id == 42
        assert f.event_id == 99999
        assert f.type == "Alarm"
        assert f.score == 85
        assert f.delta == 1.25

    def test_string_frame_id(self):
        f = Frame(frame_id="snapshot", event_id=100)
        assert f.frame_id == "snapshot"

    def test_frame_id_alarm(self):
        f = Frame(frame_id="alarm", event_id=100, type="Alarm")
        assert f.frame_id == "alarm"
        assert f.type == "Alarm"


# ===================================================================
# TestEvent
# ===================================================================

class TestEvent:
    """Tests for the Event dataclass."""

    def test_creation_minimal(self):
        ev = Event(id=12345)
        assert ev.id == 12345
        assert ev.name == ""
        assert ev.monitor_id == 0
        assert ev.cause == ""
        assert ev.notes == ""
        assert ev.start_time is None
        assert ev.end_time is None
        assert ev.length == 0.0
        assert ev.frames == 0
        assert ev.alarm_frames == 0
        assert ev.max_score == 0
        assert ev.max_score_frame_id is None
        assert ev.storage_path == ""

    def test_from_api_dict_realistic(self):
        """Test from_api_dict with a realistic ZM API response dict."""
        api_data = {
            "Event": {
                "Id": "12345",
                "Name": "Event 12345",
                "MonitorId": "2",
                "Cause": "Motion",
                "Notes": "person:97% car:85% detected by yolov4",
                "StartTime": "2024-03-15 10:30:00",
                "EndTime": "2024-03-15 10:31:30",
                "Length": "90.5",
                "Frames": "270",
                "AlarmFrames": "45",
                "MaxScore": "97",
                "MaxScoreFrameId": "135",
                "StoragePath": "/var/cache/zoneminder/events/2/2024-03-15/12345",
            }
        }

        ev = Event.from_api_dict(api_data)
        assert ev.id == 12345
        assert ev.name == "Event 12345"
        assert ev.monitor_id == 2
        assert ev.cause == "Motion"
        assert "person:97%" in ev.notes
        assert ev.start_time == datetime(2024, 3, 15, 10, 30, 0)
        assert ev.end_time == datetime(2024, 3, 15, 10, 31, 30)
        assert ev.length == 90.5
        assert ev.frames == 270
        assert ev.alarm_frames == 45
        assert ev.max_score == 97
        assert ev.max_score_frame_id == 135
        assert "12345" in ev.storage_path

    def test_from_api_dict_flat(self):
        """Test from_api_dict when data is flat (no wrapping 'Event' key)."""
        flat_data = {
            "Id": "999",
            "Name": "Test Event",
            "MonitorId": "1",
            "Cause": "Forced",
            "StartTime": "2024-01-01 00:00:00",
            "Length": "10",
            "Frames": "30",
            "AlarmFrames": "5",
            "MaxScore": "50",
        }
        ev = Event.from_api_dict(flat_data)
        assert ev.id == 999
        assert ev.name == "Test Event"
        assert ev.cause == "Forced"

    def test_from_api_dict_missing_fields(self):
        """Test from_api_dict handles missing optional fields gracefully."""
        api_data = {"Event": {"Id": "1"}}
        ev = Event.from_api_dict(api_data)
        assert ev.id == 1
        assert ev.name == ""
        assert ev.start_time is None
        assert ev.end_time is None
        assert ev.max_score_frame_id is None

    def test_from_api_dict_no_max_score_frame(self):
        api_data = {"Event": {"Id": "42", "MaxScoreFrameId": None}}
        ev = Event.from_api_dict(api_data)
        assert ev.max_score_frame_id is None

    def test_from_api_dict_iso_datetime(self):
        """Test datetime parsing with ISO format (timezone-aware)."""
        api_data = {
            "Event": {
                "Id": "1",
                "StartTime": "2024-06-15T14:30:00+0000",
            }
        }
        ev = Event.from_api_dict(api_data)
        assert ev.start_time is not None
        assert ev.start_time.year == 2024
        assert ev.start_time.month == 6
        assert ev.start_time.day == 15


# ===================================================================
# TestMonitor
# ===================================================================

class TestMonitor:
    """Tests for the Monitor dataclass."""

    def test_creation_defaults(self):
        m = Monitor(id=1)
        assert m.id == 1
        assert m.name == ""
        assert m.function == ""
        assert m.enabled is True
        assert m.width == 0
        assert m.height == 0
        assert m.type == ""
        assert m.zones == []
        assert isinstance(m.status, MonitorStatus)

    def test_from_api_dict_realistic(self):
        api_data = {
            "Monitor": {
                "Id": "3",
                "Name": "Front Door",
                "Function": "Modect",
                "Enabled": "1",
                "Width": "1920",
                "Height": "1080",
                "Type": "Ffmpeg",
            },
            "Monitor_Status": {
                "Status": "Connected",
                "CaptureFPS": "15.23",
                "Capturing": "Capturing",
            },
        }

        m = Monitor.from_api_dict(api_data)
        assert m.id == 3
        assert m.name == "Front Door"
        assert m.function == "Modect"
        assert m.enabled is True
        assert m.width == 1920
        assert m.height == 1080
        assert m.type == "Ffmpeg"
        assert m.status.state == "Connected"
        assert m.status.fps == pytest.approx(15.23)
        assert m.status.capturing == "Capturing"

    def test_from_api_dict_disabled_monitor(self):
        api_data = {
            "Monitor": {
                "Id": "5",
                "Name": "Back Yard",
                "Enabled": "0",
            },
        }
        m = Monitor.from_api_dict(api_data)
        assert m.enabled is False

    def test_from_api_dict_missing_status(self):
        api_data = {
            "Monitor": {
                "Id": "1",
                "Name": "Test",
            },
        }
        m = Monitor.from_api_dict(api_data)
        assert m.status.state == ""
        assert m.status.fps == 0.0

    def test_from_api_dict_flat(self):
        """Test flat dict without wrapping 'Monitor' key."""
        flat = {
            "Id": "10",
            "Name": "Flat Monitor",
            "Function": "Monitor",
            "Enabled": "1",
            "Width": "640",
            "Height": "480",
            "Type": "Local",
        }
        m = Monitor.from_api_dict(flat)
        assert m.id == 10
        assert m.name == "Flat Monitor"

    def test_from_api_dict_null_fps(self):
        """Test that null/empty CaptureFPS defaults to 0."""
        api_data = {
            "Monitor": {"Id": "1"},
            "Monitor_Status": {"CaptureFPS": None},
        }
        m = Monitor.from_api_dict(api_data)
        assert m.status.fps == 0.0


# ===================================================================
# TestMonitorStatus
# ===================================================================

class TestMonitorStatus:
    """Tests for the MonitorStatus dataclass."""

    def test_defaults(self):
        ms = MonitorStatus()
        assert ms.state == ""
        assert ms.fps == 0.0
        assert ms.analysis_fps == 0.0
        assert ms.bandwidth == 0
        assert ms.capturing == "None"

    def test_custom_values(self):
        ms = MonitorStatus(state="Alarm", fps=25.0, capturing="Capturing")
        assert ms.state == "Alarm"
        assert ms.fps == 25.0
        assert ms.capturing == "Capturing"

    def test_analysis_fps_and_bandwidth(self):
        """Ref: ZoneMinder/pyzm#53"""
        ms = MonitorStatus(
            state="Connected", fps=15.0,
            analysis_fps=14.5, bandwidth=52095,
            capturing="Capturing",
        )
        assert ms.analysis_fps == 14.5
        assert ms.bandwidth == 52095

    def test_from_api_dict_with_status_fields(self):
        """Ref: ZoneMinder/pyzm#53 -- Monitor_Status API fields."""
        api_data = {
            "Monitor": {"Id": "1", "Name": "Test"},
            "Monitor_Status": {
                "Status": "Connected",
                "CaptureFPS": "15.23",
                "AnalysisFPS": "14.80",
                "CaptureBandwidth": "52095",
                "Capturing": "Capturing",
            },
        }
        m = Monitor.from_api_dict(api_data)
        assert m.status.fps == pytest.approx(15.23)
        assert m.status.analysis_fps == pytest.approx(14.80)
        assert m.status.bandwidth == 52095

    def test_from_api_dict_null_analysis_fps(self):
        """Null AnalysisFPS defaults to 0."""
        api_data = {
            "Monitor": {"Id": "1"},
            "Monitor_Status": {"AnalysisFPS": None, "CaptureBandwidth": None},
        }
        m = Monitor.from_api_dict(api_data)
        assert m.status.analysis_fps == 0.0
        assert m.status.bandwidth == 0


# ===================================================================
# TestZoneIgnorePattern
# ===================================================================

class TestZoneIgnorePattern:
    """Tests for Zone.ignore_pattern field. Ref: ZoneMinder/pyzm#37"""

    def test_default_none(self):
        z = Zone(name="test", points=[(0, 0), (10, 10)])
        assert z.ignore_pattern is None

    def test_as_dict_includes_ignore_pattern(self):
        z = Zone(name="driveway", points=[(0, 0)], ignore_pattern="(car|truck)")
        d = z.as_dict()
        assert d["ignore_pattern"] == "(car|truck)"

    def test_as_dict_ignore_pattern_none(self):
        z = Zone(name="test", points=[(0, 0)])
        d = z.as_dict()
        assert d["ignore_pattern"] is None


# ===================================================================
# TestRaw -- raw() method on all models
# ===================================================================

class TestRaw:
    """Tests for the raw() escape hatch on all ZM model dataclasses."""

    def test_monitor_raw_from_api_dict(self):
        api_data = {
            "Monitor": {
                "Id": "3", "Name": "Front Door", "Function": "Modect",
                "Enabled": "1", "Width": "1920", "Height": "1080",
                "Type": "Ffmpeg", "Path": "rtsp://cam/stream",
                "Protocol": "rtsp", "MaxFPS": "30",
            },
            "Monitor_Status": {
                "Status": "Connected", "CaptureFPS": "15.23",
            },
        }
        m = Monitor.from_api_dict(api_data)
        assert m.raw() is api_data
        assert m.raw()["Monitor"]["Path"] == "rtsp://cam/stream"

    def test_monitor_status_raw_from_api_dict(self):
        status_data = {"Status": "Connected", "CaptureFPS": "15.23"}
        api_data = {
            "Monitor": {"Id": "1"},
            "Monitor_Status": status_data,
        }
        m = Monitor.from_api_dict(api_data)
        assert m.status.raw() is status_data

    def test_monitor_raw_default_empty(self):
        m = Monitor(id=1)
        assert m.raw() == {}

    def test_event_raw_from_api_dict(self):
        api_data = {
            "Event": {
                "Id": "99", "Name": "Event 99", "MonitorId": "2",
                "Cause": "Motion", "StorageId": "1",
                "DiskSpace": "12345678",
            },
        }
        ev = Event.from_api_dict(api_data)
        assert ev.raw() is api_data
        assert ev.raw()["Event"]["DiskSpace"] == "12345678"

    def test_event_raw_default_empty(self):
        ev = Event(id=1)
        assert ev.raw() == {}

    def test_frame_raw(self):
        raw_entry = {
            "Frame": {
                "FrameId": "5", "EventId": "100", "Type": "Alarm",
                "Score": "80", "Delta": "1.5", "TimeStamp": "2024-01-01 00:00:05",
            },
        }
        f = Frame(frame_id=5, event_id=100, type="Alarm", score=80, delta=1.5, _raw=raw_entry)
        assert f.raw() is raw_entry
        assert f.raw()["Frame"]["TimeStamp"] == "2024-01-01 00:00:05"

    def test_frame_raw_default_empty(self):
        f = Frame(frame_id=1, event_id=1)
        assert f.raw() == {}

    def test_zone_raw(self):
        raw_entry = {
            "Zone": {
                "Id": "1", "Name": "All", "MonitorId": "1",
                "Type": "Active", "Coords": "0,0 639,0 639,479 0,479",
                "AlarmRGB": "16711680",
            },
        }
        z = Zone(name="All", points=[(0, 0), (639, 0), (639, 479), (0, 479)], _raw=raw_entry)
        assert z.raw() is raw_entry
        assert z.raw()["Zone"]["AlarmRGB"] == "16711680"

    def test_zone_raw_default_empty(self):
        z = Zone(name="test", points=[])
        assert z.raw() == {}

    def test_ptz_capabilities_raw_from_api_dict(self):
        ctrl_data = {
            "Id": "1", "Name": "Amcrest", "Type": "Ffmpeg",
            "CanMove": "1", "CanZoom": "1", "CanPan": "1", "CanTilt": "1",
            "HasPresets": "1", "NumPresets": "8",
        }
        ptz = PTZCapabilities.from_api_dict(ctrl_data)
        assert ptz.raw() is ctrl_data
        assert ptz.raw()["Name"] == "Amcrest"

    def test_ptz_capabilities_raw_default_empty(self):
        ptz = PTZCapabilities()
        assert ptz.raw() == {}

    def test_monitor_status_raw_default_empty(self):
        ms = MonitorStatus()
        assert ms.raw() == {}

    def test_raw_not_in_repr(self):
        """_raw should be excluded from repr output."""
        api_data = {"Monitor": {"Id": "1", "Name": "Test"}, "Monitor_Status": {}}
        m = Monitor.from_api_dict(api_data)
        assert "_raw" not in repr(m)

    def test_raw_not_in_equality(self):
        """_raw should not affect equality comparison."""
        m1 = Monitor(id=1, name="Test", _raw={"Monitor": {"Id": "1"}})
        m2 = Monitor(id=1, name="Test", _raw={"different": "data"})
        assert m1 == m2
