"""Tests for pyzm.train.zm_browser."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyzm.models.zm import Event, Frame, Monitor
from pyzm.train.dataset import YOLODataset
from pyzm.train.verification import (
    ImageVerification,
    VerificationStore,
)
from pyzm.train.zm_browser import (
    _all_alarm_frames,
    _custom_frames,
    _fetch_event_thumbnails,
    _fetch_frame_image,
    _get_cached_thumbnails,
    _import_frames,
    _initial_key_frames,
    _load_zm_creds,
    _placeholder_image,
    _portal_url,
    _save_zm_creds,
    _zm_disconnect,
)


@pytest.fixture
def project(tmp_path: Path):
    ds = YOLODataset(project_dir=tmp_path, classes=[])
    ds.init_project()
    store = VerificationStore(tmp_path)
    return ds, store


@pytest.fixture
def args():
    return argparse.Namespace(base_path="/tmp/models", processor="cpu")


@pytest.fixture
def fake_monitors():
    return [
        Monitor(id=1, name="Front Door", function="Modect", width=1920, height=1080),
        Monitor(id=2, name="Back Yard", function="Monitor", width=1280, height=720),
    ]


@pytest.fixture
def fake_events():
    return [
        Event(
            id=54321, name="Event 54321", monitor_id=1,
            start_time=datetime(2026, 2, 15, 8, 30),
            length=90.0, frames=270, alarm_frames=45, max_score=97,
        ),
        Event(
            id=54320, name="Event 54320", monitor_id=1,
            start_time=datetime(2026, 2, 15, 8, 15),
            length=45.0, frames=90, alarm_frames=12, max_score=65,
        ),
    ]


@pytest.fixture
def fake_frame_meta():
    return [
        Frame(frame_id=1, event_id=54321, type="Normal", score=0),
        Frame(frame_id=5, event_id=54321, type="Alarm", score=85),
        Frame(frame_id=10, event_id=54321, type="Alarm", score=92),
        Frame(frame_id=15, event_id=54321, type="Normal", score=0),
        Frame(frame_id=20, event_id=54321, type="Alarm", score=78),
    ]


# ---------------------------------------------------------------------------
# _zm_disconnect
# ---------------------------------------------------------------------------

class TestZmDisconnect:
    def test_clears_zm_keys(self):
        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {
                "zm_connected": True,
                "zm_client": MagicMock(),
                "zm_monitors": [],
                "zm_url": "http://x",
                "other_key": "keep",
            }
            _zm_disconnect()
            assert "zm_connected" not in mock_st.session_state
            assert "zm_client" not in mock_st.session_state
            assert "zm_monitors" not in mock_st.session_state
            assert "zm_url" not in mock_st.session_state
            assert "other_key" in mock_st.session_state


# ---------------------------------------------------------------------------
# _save_zm_creds / _load_zm_creds
# ---------------------------------------------------------------------------

class TestZmCreds:
    def test_save_and_load_roundtrip(self, project):
        ds, _store = project
        _save_zm_creds(ds.project_dir, "https://zm.local/zm", "admin", False)

        loaded = _load_zm_creds(ds.project_dir)
        assert loaded["url"] == "https://zm.local/zm"
        assert loaded["user"] == "admin"
        assert loaded["verify_ssl"] is False

    def test_save_preserves_existing_project_json(self, project):
        ds, _store = project
        import json
        meta_path = ds.project_dir / "project.json"
        meta = json.loads(meta_path.read_text())
        assert "classes" in meta

        _save_zm_creds(ds.project_dir, "http://x", "u", True)

        meta = json.loads(meta_path.read_text())
        assert "classes" in meta
        assert meta["zm_connection"]["url"] == "http://x"

    def test_password_not_saved(self, project):
        """Password must never be persisted in project.json."""
        import json
        ds, _store = project
        _save_zm_creds(ds.project_dir, "https://zm.local/zm", "admin", True)

        raw = json.loads((ds.project_dir / "project.json").read_text())
        zm_conn = raw["zm_connection"]
        assert "password" not in zm_conn
        assert "url" in zm_conn
        assert "user" in zm_conn
        assert "verify_ssl" in zm_conn

    def test_load_returns_empty_when_no_creds(self, project):
        ds, _store = project
        loaded = _load_zm_creds(ds.project_dir)
        assert loaded == {}

    def test_load_returns_empty_for_missing_dir(self, tmp_path):
        loaded = _load_zm_creds(tmp_path / "nonexistent")
        assert loaded == {}


# ---------------------------------------------------------------------------
# _import_frames
# ---------------------------------------------------------------------------

class TestImportFrames:
    def test_import_saves_images_and_verifications(self, project, args):
        ds, store = project

        mock_zm = MagicMock()
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_zm.event.return_value.extract_frames.return_value = (
            [(5, fake_img.copy()), (10, fake_img.copy())],
            {"original": (100, 100), "resized": None},
        )

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"base_model": "yolo11s", "model_class_names": []}
            mock_st.progress.return_value = MagicMock()
            mock_st.toast = MagicMock()

            _import_frames(ds, store, args, mock_zm, 54321, [5, 10])

        images = ds.staged_images()
        assert len(images) == 2

        for img in images:
            iv = store.get(img.name)
            assert iv is not None
            assert iv.fully_reviewed is False

        names = [img.name for img in images]
        assert any("event54321_frame5" in n for n in names)
        assert any("event54321_frame10" in n for n in names)

    def test_import_saves_frames_without_detection(self, project, args):
        """Frames are imported with empty detections (detection deferred to review)."""
        ds, store = project

        mock_zm = MagicMock()
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_zm.event.return_value.extract_frames.return_value = (
            [(1, fake_img)],
            {"original": (100, 100), "resized": None},
        )

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {
                "base_model": "yolo11s",
                "model_class_names": ["person", "car"],
            }
            mock_st.progress.return_value = MagicMock()
            mock_st.toast = MagicMock()

            _import_frames(ds, store, args, mock_zm, 99, [1])

        images = ds.staged_images()
        assert len(images) == 1
        iv = store.get(images[0].name)
        assert len(iv.detections) == 0
        assert iv.fully_reviewed is False

    def test_import_handles_fetch_error(self, project, args):
        ds, store = project

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.side_effect = ConnectionError("timeout")

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"base_model": "yolo11s", "model_class_names": []}
            mock_st.progress.return_value = MagicMock()
            mock_st.error = MagicMock()

            _import_frames(ds, store, args, mock_zm, 99, [1])

        mock_st.error.assert_called_once()
        assert ds.staged_images() == []

    def test_import_with_no_frames_returned(self, project, args):
        ds, store = project

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([], {"original": None, "resized": None})

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"base_model": "yolo11s", "model_class_names": []}
            mock_st.progress.return_value = MagicMock()
            mock_st.toast = MagicMock()

            _import_frames(ds, store, args, mock_zm, 99, [1, 2])

        assert ds.staged_images() == []

    def test_import_stream_config_uses_frame_ids(self, project, args):
        ds, store = project

        mock_zm = MagicMock()
        mock_ev = mock_zm.event.return_value
        mock_ev.extract_frames.return_value = ([], {"original": None, "resized": None})

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"base_model": "yolo11s", "model_class_names": []}
            mock_st.progress.return_value = MagicMock()
            mock_st.toast = MagicMock()

            _import_frames(ds, store, args, mock_zm, 100, [5, 10, 15])

        mock_zm.event.assert_called_once_with(100)
        call_args = mock_ev.extract_frames.call_args
        sc = call_args[1]["stream_config"]
        assert sc.frame_set == ["5", "10", "15"]
        assert sc.resize is None

    def test_import_saves_store_to_disk(self, project, args):
        ds, store = project

        mock_zm = MagicMock()
        fake_img = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_zm.event.return_value.extract_frames.return_value = (
            [(1, fake_img)],
            {"original": (50, 50), "resized": None},
        )

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"base_model": "yolo11s", "model_class_names": []}
            mock_st.progress.return_value = MagicMock()
            mock_st.toast = MagicMock()

            _import_frames(ds, store, args, mock_zm, 42, [1])

        vfile = project[0].project_dir / "verifications.json"
        assert vfile.exists()

        store2 = VerificationStore(project[0].project_dir)
        assert len(store2.all_images()) == 1


# ---------------------------------------------------------------------------
# _initial_key_frames / _all_alarm_frames / _custom_frames
# ---------------------------------------------------------------------------

class TestInitialKeyFrames:
    def test_returns_first_alarm_only_when_no_snapshot(self, fake_frame_meta):
        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"zm_events": []}
            result = _initial_key_frames(fake_frame_meta, 54321)

        assert len(result) == 1
        assert result[0].frame_id == 5

    def test_returns_first_alarm_and_snapshot(self, fake_frame_meta):
        ev = Event(id=54321, max_score_frame_id=15)
        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"zm_events": [ev]}
            result = _initial_key_frames(fake_frame_meta, 54321)

        fids = [f.frame_id for f in result]
        assert fids == [5, 15]

    def test_no_duplicates_when_snapshot_is_first_alarm(self, fake_frame_meta):
        ev = Event(id=54321, max_score_frame_id=5)
        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"zm_events": [ev]}
            result = _initial_key_frames(fake_frame_meta, 54321)

        assert len(result) == 1
        assert result[0].frame_id == 5

    def test_empty_input(self):
        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"zm_events": []}
            assert _initial_key_frames([], 99) == []


class TestAllAlarmFrames:
    def test_returns_all_alarm_type(self, fake_frame_meta):
        result = _all_alarm_frames(fake_frame_meta)
        fids = [f.frame_id for f in result]
        assert fids == [5, 10, 20]

    def test_empty_input(self):
        assert _all_alarm_frames([]) == []


class TestCustomFrames:
    def test_returns_matching_frames(self, fake_frame_meta):
        result = _custom_frames(fake_frame_meta, [1, 10, 999])
        fids = [f.frame_id for f in result]
        assert 1 in fids
        assert 10 in fids
        assert 999 not in fids

    def test_empty_ids(self, fake_frame_meta):
        assert _custom_frames(fake_frame_meta, []) == []


# ---------------------------------------------------------------------------
# _portal_url
# ---------------------------------------------------------------------------

class TestPortalUrl:
    def test_strips_api_suffix(self):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm/api"
        assert _portal_url(mock_zm) == "https://zm.example.com/zm"

    def test_no_api_suffix_unchanged(self):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm"
        assert _portal_url(mock_zm) == "https://zm.example.com/zm"


# ---------------------------------------------------------------------------
# _fetch_frame_image
# ---------------------------------------------------------------------------

class TestFetchFrameImage:
    def test_returns_bytes_on_success(self):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm/api"
        fake_resp = MagicMock()
        fake_resp.content = b"\xff\xd8\xff\xe0fake-jpeg-data"
        mock_zm.api.request.return_value = fake_resp

        result = _fetch_frame_image(mock_zm, 123, 5, width=160)
        assert result == b"\xff\xd8\xff\xe0fake-jpeg-data"

        call_url = mock_zm.api.request.call_args[0][0]
        assert "view=image" in call_url
        assert "eid=123" in call_url
        assert "fid=5" in call_url
        assert "width=160" in call_url

    def test_returns_none_on_error(self):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm/api"
        mock_zm.api.request.side_effect = ConnectionError("fail")

        result = _fetch_frame_image(mock_zm, 123, 5)
        assert result is None

    def test_no_width_param_when_none(self):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm/api"
        fake_resp = MagicMock()
        fake_resp.content = b"data"
        mock_zm.api.request.return_value = fake_resp

        _fetch_frame_image(mock_zm, 123, 5, width=None)
        call_url = mock_zm.api.request.call_args[0][0]
        assert "width=" not in call_url


# ---------------------------------------------------------------------------
# _get_cached_thumbnails
# ---------------------------------------------------------------------------

class TestGetCachedThumbnails:
    def test_fetches_and_caches(self, fake_frame_meta):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm/api"
        fake_resp = MagicMock()
        fake_resp.content = b"thumb-data"
        mock_zm.api.request.return_value = fake_resp

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {}
            thumbs = _get_cached_thumbnails(mock_zm, 54321, fake_frame_meta)

        assert len(thumbs) == 5
        for fid in [1, 5, 10, 15, 20]:
            assert thumbs[fid] == b"thumb-data"

        assert "zm_thumbs_54321" in mock_st.session_state

    def test_uses_cache_on_second_call(self, fake_frame_meta):
        """When all frames are already cached, no API requests are made."""
        mock_zm = MagicMock()
        cached = {fm.frame_id: b"cached-thumb" for fm in fake_frame_meta}

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"zm_thumbs_54321": cached}
            thumbs = _get_cached_thumbnails(mock_zm, 54321, fake_frame_meta)

        mock_zm.api.request.assert_not_called()
        assert thumbs is cached


# ---------------------------------------------------------------------------
# _fetch_event_thumbnails
# ---------------------------------------------------------------------------

class TestFetchEventThumbnails:
    def test_fetches_for_all_events(self, fake_events):
        mock_zm = MagicMock()
        mock_zm.api.api_url = "https://zm.example.com/zm/api"
        fake_resp = MagicMock()
        fake_resp.content = b"event-thumb"
        mock_zm.api.request.return_value = fake_resp

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {}
            thumbs = _fetch_event_thumbnails(mock_zm, fake_events)

        assert 54321 in thumbs
        assert 54320 in thumbs

    def test_uses_cache(self, fake_events):
        mock_zm = MagicMock()
        cached = {54321: b"cached", 54320: b"cached2"}

        with patch("pyzm.train.zm_browser.st") as mock_st:
            mock_st.session_state = {"zm_event_thumbs": cached}
            thumbs = _fetch_event_thumbnails(mock_zm, fake_events)

        mock_zm.api.request.assert_not_called()
        assert thumbs is cached


# ---------------------------------------------------------------------------
# _placeholder_image
# ---------------------------------------------------------------------------

class TestPlaceholderImage:
    def test_returns_bytes(self):
        result = _placeholder_image()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_same_bytes_on_second_call(self):
        a = _placeholder_image()
        b = _placeholder_image()
        assert a is b
