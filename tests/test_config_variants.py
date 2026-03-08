"""Comprehensive config variant tests for local and remote modes.

Tests config parsing, option threading, and correctness of recent fixes
across DetectorConfig, Detector, ModelPipeline, AuthManager, and DB
connection code. No real models, ZoneMinder, or database connections needed.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import (
    DetectorConfig,
    FrameStrategy,
    MatchStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
    Processor,
    StreamConfig,
    TypeOverrides,
    _bool,
)
from pyzm.models.detection import BBox, Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ml_options(
    *,
    model_sequence: str = "object",
    global_general: dict | None = None,
    sections: dict | None = None,
):
    """Build an ml_options dict. *sections* maps type name -> {general, sequence}."""
    gen = {"model_sequence": model_sequence}
    if global_general:
        gen.update(global_general)

    opts: dict = {"general": gen}
    if sections:
        opts.update(sections)
    else:
        # Default: single object model
        opts["object"] = {
            "general": {},
            "sequence": [{"name": "test", "object_weights": "/tmp/w.onnx"}],
        }
    return opts


# ===========================================================================
# 1. TestDetectorConfigVariants
# ===========================================================================

class TestDetectorConfigVariants:
    """Tests for DetectorConfig.from_dict() with various YAML-like config structures."""

    def test_single_object_model_all_options(self):
        ml = _ml_options(sections={
            "object": {
                "general": {},
                "sequence": [{
                    "name": "yolo11s",
                    "object_weights": "/models/yolo11s.onnx",
                    "object_config": "/models/yolo11s.cfg",
                    "object_labels": "/models/coco.names",
                    "object_framework": "opencv",
                    "object_processor": "gpu",
                    "object_min_confidence": 0.5,
                    "model_width": 640,
                    "model_height": 640,
                    "max_detection_size": "80%",
                }],
            },
        }, global_general={"pattern": "(person|car)"})

        cfg = DetectorConfig.from_dict(ml)
        assert len(cfg.models) == 1
        m = cfg.models[0]
        assert m.name == "yolo11s"
        assert m.weights == "/models/yolo11s.onnx"
        assert m.config == "/models/yolo11s.cfg"
        assert m.labels == "/models/coco.names"
        assert m.framework == ModelFramework.OPENCV
        assert m.processor == Processor.GPU
        assert m.min_confidence == 0.5
        assert m.model_width == 640
        assert m.model_height == 640
        assert m.max_detection_size == "80%"
        assert m.pattern == "(person|car)"

    def test_multi_model_sequence_object_face(self):
        ml = _ml_options(
            model_sequence="object,face",
            sections={
                "object": {
                    "general": {"same_model_sequence_strategy": "most"},
                    "sequence": [
                        {"name": "yolo", "object_weights": "/w.onnx"},
                    ],
                },
                "face": {
                    "general": {"same_model_sequence_strategy": "first"},
                    "sequence": [
                        {"name": "dlib", "face_detection_framework": "dlib"},
                    ],
                },
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        assert len(cfg.models) == 2
        assert cfg.models[0].type == ModelType.OBJECT
        assert cfg.models[0].name == "yolo"
        assert cfg.models[1].type == ModelType.FACE
        assert cfg.models[1].name == "dlib"
        assert cfg.models[1].framework == ModelFramework.FACE_DLIB
        # Per-type overrides populated
        assert cfg.type_overrides[ModelType.OBJECT].match_strategy == MatchStrategy.MOST
        assert cfg.type_overrides[ModelType.FACE].match_strategy == MatchStrategy.FIRST

    def test_alpr_model_extended_options(self):
        ml = _ml_options(
            model_sequence="alpr",
            sections={
                "alpr": {
                    "general": {},
                    "sequence": [{
                        "alpr_service": "plate_recognizer",
                        "alpr_key": "tok_abc",
                        "alpr_url": "https://api.platerecognizer.com/v1/plate-reader/",
                        "platerec_stats": "yes",
                        "platerec_regions": ["us", "eu"],
                        "alpr_api_type": "cloud",
                    }],
                },
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        m = cfg.models[0]
        assert m.type == ModelType.ALPR
        assert m.alpr_service == "plate_recognizer"
        assert m.alpr_key == "tok_abc"
        # Extended options flow into options dict
        assert m.options["platerec_stats"] == "yes"
        assert m.options["platerec_regions"] == ["us", "eu"]
        assert m.options["alpr_api_type"] == "cloud"

    @pytest.mark.parametrize("val,expected", [
        (True, True),
        ("yes", True),
        (False, False),
        ("no", False),
    ])
    def test_enabled_boolean_handling(self, val, expected):
        ml = _ml_options(sections={
            "object": {"general": {}, "sequence": [{"enabled": val}]},
        })
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].enabled is expected

    @pytest.mark.parametrize("val,expected", [
        (True, True),
        ("yes", True),
        (False, False),
        ("no", False),
    ])
    def test_match_past_detections_boolean_handling(self, val, expected):
        ml = _ml_options(global_general={"match_past_detections": val})
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.match_past_detections is expected

    @pytest.mark.parametrize("val,expected", [
        (True, True),
        ("yes", True),
        (False, False),
    ])
    def test_disable_locks_boolean_handling(self, val, expected):
        ml = _ml_options(sections={
            "object": {"general": {}, "sequence": [{"disable_locks": val}]},
        })
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].disable_locks is expected

    def test_disable_locks_from_global(self):
        ml = _ml_options(
            global_general={"disable_locks": "yes"},
            sections={
                "object": {"general": {}, "sequence": [{}]},
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].disable_locks is True

    def test_max_lock_wait_sequence_overrides_global(self):
        ml = _ml_options(
            global_general={"max_lock_wait": 60, "max_processes": 2},
            sections={
                "object": {
                    "general": {},
                    "sequence": [{"max_lock_wait": 30, "max_processes": 4}],
                },
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].max_lock_wait == 30
        assert cfg.models[0].max_processes == 4

    def test_max_lock_wait_falls_back_to_global(self):
        ml = _ml_options(
            global_general={"max_lock_wait": 90, "max_processes": 3},
            sections={
                "object": {"general": {}, "sequence": [{}]},
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].max_lock_wait == 90
        assert cfg.models[0].max_processes == 3

    def test_monitor_id_threaded(self):
        ml = _ml_options(global_general={"monitor_id": "7"})
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.monitor_id == "7"

    def test_image_path_default(self):
        cfg = DetectorConfig.from_dict(_ml_options())
        assert cfg.image_path == "/var/lib/zmeventnotification/images"

    def test_image_path_custom(self):
        ml = _ml_options(global_general={"image_path": "/custom/path"})
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.image_path == "/custom/path"

    def test_per_type_overrides(self):
        ml = _ml_options(sections={
            "object": {
                "general": {
                    "same_model_sequence_strategy": "most_unique",
                    "match_past_detections": True,
                    "past_det_max_diff_area": "10%",
                    "ignore_past_detection_labels": ["bird"],
                    "aliases": [["car", "truck"]],
                },
                "sequence": [{"name": "y"}],
            },
        })
        cfg = DetectorConfig.from_dict(ml)
        tov = cfg.type_overrides[ModelType.OBJECT]
        assert tov.match_strategy == MatchStrategy.MOST_UNIQUE
        assert tov.match_past_detections is True
        assert tov.past_det_max_diff_area == "10%"
        assert tov.ignore_past_detection_labels == ["bird"]
        assert tov.aliases == [["car", "truck"]]

    @pytest.mark.parametrize("strat", [
        "first", "first_new", "most", "most_unique", "most_models",
    ])
    def test_frame_strategy_variants(self, strat):
        ml = _ml_options(global_general={"frame_strategy": strat})
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.frame_strategy == FrameStrategy(strat)

    def test_pattern_global_vs_section_vs_model(self):
        """Section-level pattern overrides global; model inherits section."""
        ml = _ml_options(
            global_general={"pattern": "(person)"},
            sections={
                "object": {
                    "general": {"pattern": "(car|truck)"},
                    "sequence": [{"name": "y"}],
                },
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        # Section pattern wins over global
        assert cfg.models[0].pattern == "(car|truck)"

    def test_pattern_model_inherits_global_when_no_section(self):
        ml = _ml_options(
            global_general={"pattern": "(dog|cat)"},
            sections={
                "object": {
                    "general": {},
                    "sequence": [{"name": "y"}],
                },
            },
        )
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].pattern == "(dog|cat)"

    def test_max_detection_size_cascade(self):
        """max_detection_size: model-level > section > global."""
        # Model-level set
        ml_model = _ml_options(
            global_general={"max_detection_size": "50%"},
            sections={
                "object": {
                    "general": {"max_detection_size": "70%"},
                    "sequence": [{"name": "y", "max_detection_size": "90%"}],
                },
            },
        )
        cfg = DetectorConfig.from_dict(ml_model)
        assert cfg.models[0].max_detection_size == "90%"

        # No model-level: section wins
        ml_section = _ml_options(
            global_general={"max_detection_size": "50%"},
            sections={
                "object": {
                    "general": {"max_detection_size": "70%"},
                    "sequence": [{"name": "y"}],
                },
            },
        )
        cfg2 = DetectorConfig.from_dict(ml_section)
        assert cfg2.models[0].max_detection_size == "70%"

        # No model or section: None at model level (global only applies to DetectorConfig)
        ml_global = _ml_options(
            global_general={"max_detection_size": "50%"},
            sections={
                "object": {
                    "general": {},
                    "sequence": [{"name": "y"}],
                },
            },
        )
        cfg3 = DetectorConfig.from_dict(ml_global)
        # Global max_detection_size goes to DetectorConfig.max_detection_size, not model
        assert cfg3.max_detection_size == "50%"
        assert cfg3.models[0].max_detection_size is None

    def test_per_label_past_det_area_overrides(self):
        ml = _ml_options(global_general={
            "past_det_max_diff_area": "5%",
            "car_past_det_max_diff_area": "15%",
            "person_past_det_max_diff_area": "20%",
        })
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.past_det_max_diff_area_labels["car"] == "15%"
        assert cfg.past_det_max_diff_area_labels["person"] == "20%"


# ===========================================================================
# 2. TestDetectorFromDict
# ===========================================================================

class TestDetectorFromDict:
    """Tests for Detector.from_dict() config threading."""

    def test_gateway_settings_threaded(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options(global_general={
            "ml_gateway": "http://gpu:5000",
            "ml_user": "admin",
            "ml_password": "secret",
            "ml_timeout": 120,
            "ml_gateway_mode": "url",
        })
        det = Detector.from_dict(ml)
        assert det._gateway == "http://gpu:5000"
        assert det._gateway_mode == "url"
        assert det._gateway_timeout == 120
        assert det._gateway_username == "admin"
        assert det._gateway_password == "secret"

    def test_gateway_mode_image(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options(global_general={
            "ml_gateway": "http://gpu:5000",
            "ml_gateway_mode": "image",
        })
        det = Detector.from_dict(ml)
        assert det._gateway_mode == "image"

    def test_no_gateway_local_mode(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options()
        det = Detector.from_dict(ml)
        assert det._gateway is None

    def test_disabled_model_gets_enabled_false(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options(sections={
            "object": {
                "general": {},
                "sequence": [{"name": "y", "enabled": "no"}],
            },
        })
        det = Detector.from_dict(ml)
        assert det._config.models[0].enabled is False

    def test_monitor_id_flows_through(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options(global_general={"monitor_id": "12"})
        det = Detector.from_dict(ml)
        assert det._config.monitor_id == "12"

    def test_gateway_trailing_slash_stripped(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options(global_general={
            "ml_gateway": "http://gpu:5000/",
        })
        det = Detector.from_dict(ml)
        assert det._gateway == "http://gpu:5000"


# ===========================================================================
# 3. TestPerMonitorPastDetections
# ===========================================================================

class TestPerMonitorPastDetections:
    """Tests that the pipeline uses per-monitor past detection file paths."""

    def test_with_monitor_id_uses_mid_file(self, tmp_path):
        from pyzm.ml.pipeline import ModelPipeline

        config = DetectorConfig(
            models=[],
            match_past_detections=True,
            image_path=str(tmp_path),
            monitor_id="3",
        )
        pipeline = ModelPipeline(config)

        det = Detection(
            label="person", confidence=0.9,
            bbox=BBox(x1=10, y1=10, x2=100, y2=100),
            model_name="test",
        )

        # Patch match_past_detections to return all detections unchanged
        with patch("pyzm.ml.filters.load_past_detections", return_value=([], [])), \
             patch("pyzm.ml.filters.match_past_detections", return_value=[det]), \
             patch("pyzm.ml.filters.save_past_detections") as mock_save:
            result = pipeline._filter_past_per_type([det])

        expected_path = os.path.join(str(tmp_path), "past_detections_mid3.pkl")
        mock_save.assert_called_once_with(expected_path, [det])
        assert len(result) == 1

    def test_without_monitor_id_uses_default_file(self, tmp_path):
        from pyzm.ml.pipeline import ModelPipeline

        config = DetectorConfig(
            models=[],
            match_past_detections=True,
            image_path=str(tmp_path),
        )
        pipeline = ModelPipeline(config)

        det = Detection(
            label="car", confidence=0.85,
            bbox=BBox(x1=5, y1=5, x2=50, y2=50),
            model_name="test",
        )

        with patch("pyzm.ml.filters.load_past_detections", return_value=([], [])), \
             patch("pyzm.ml.filters.match_past_detections", return_value=[det]), \
             patch("pyzm.ml.filters.save_past_detections") as mock_save:
            result = pipeline._filter_past_per_type([det])

        expected_path = os.path.join(str(tmp_path), "past_detections.pkl")
        mock_save.assert_called_once_with(expected_path, [det])

    def test_different_monitor_ids_use_different_files(self, tmp_path):
        from pyzm.ml.pipeline import ModelPipeline

        det = Detection(
            label="person", confidence=0.9,
            bbox=BBox(x1=10, y1=10, x2=100, y2=100),
            model_name="test",
        )

        paths_used = []

        def capture_save(path, _detections):
            paths_used.append(path)

        for mid in ("1", "2"):
            config = DetectorConfig(
                models=[],
                match_past_detections=True,
                image_path=str(tmp_path),
                monitor_id=mid,
            )
            pipeline = ModelPipeline(config)
            with patch("pyzm.ml.filters.load_past_detections", return_value=([], [])), \
                 patch("pyzm.ml.filters.match_past_detections", return_value=[det]), \
                 patch("pyzm.ml.filters.save_past_detections", side_effect=capture_save):
                pipeline._filter_past_per_type([det])

        assert paths_used[0] != paths_used[1]
        assert "mid1" in paths_used[0]
        assert "mid2" in paths_used[1]


# ===========================================================================
# 4. TestAuthURLSeparator
# ===========================================================================

class TestAuthURLSeparator:
    """Test that AuthManager.apply_auth() correctly determines URL separator."""

    def _make_auth(self):
        """Create an AuthManager with legacy credentials mocked."""
        from pyzm.zm.auth import AuthManager

        session = MagicMock()
        auth = AuthManager(session=session, api_url="http://zm/zm/api", user="u", password="p")
        # Force legacy mode
        auth.auth_enabled = True
        auth._legacy_credentials = "user=x&pass=y"
        auth.api_version = "1.0"
        auth.zm_version = "1.30.0"
        return auth

    def test_url_without_query_uses_question_mark(self):
        auth = self._make_auth()
        url, _ = auth.apply_auth("http://zm/zm/index.php")
        assert "?user=x&pass=y" in url
        assert "&user=x&pass=y" not in url.split("?")[0]

    def test_url_with_query_uses_ampersand(self):
        auth = self._make_auth()
        url, _ = auth.apply_auth("http://zm/zm/index.php?view=image&eid=123")
        assert "&user=x&pass=y" in url
        # Should not have "??..."
        assert "??" not in url

    def test_auth_disabled_returns_url_unchanged(self):
        auth = self._make_auth()
        auth.auth_enabled = False
        url, _ = auth.apply_auth("http://zm/zm/index.php")
        assert url == "http://zm/zm/index.php"


# ===========================================================================
# 5. TestDBPortParsing
# ===========================================================================

class TestDBPortParsing:
    """Test that get_zm_db handles malformed port gracefully.

    ``get_zm_db`` does ``import mysql.connector`` lazily inside the function,
    so we patch ``mysql.connector.connect`` on the real module to intercept
    connection attempts without touching the database.
    """

    def test_normal_host_port(self):
        from pyzm.zm.db import get_zm_db

        mock_connect = MagicMock(return_value=MagicMock())
        with patch("pyzm.zm.db._read_zm_conf", return_value={
            "user": "zmuser", "password": "zmpass",
            "host": "localhost:3306", "database": "zm",
        }), patch("mysql.connector.connect", mock_connect):
            get_zm_db()

        mock_connect.assert_called_once()
        kw = mock_connect.call_args.kwargs
        assert kw["host"] == "localhost"
        assert kw["port"] == 3306

    def test_malformed_port_falls_back_to_3306(self):
        from pyzm.zm.db import get_zm_db

        mock_connect = MagicMock(return_value=MagicMock())
        with patch("pyzm.zm.db._read_zm_conf", return_value={
            "user": "zmuser", "password": "zmpass",
            "host": "localhost:notaport", "database": "zm",
        }), patch("mysql.connector.connect", mock_connect):
            get_zm_db()

        kw = mock_connect.call_args.kwargs
        assert kw["port"] == 3306

    def test_unix_socket(self):
        from pyzm.zm.db import get_zm_db

        mock_connect = MagicMock(return_value=MagicMock())
        with patch("pyzm.zm.db._read_zm_conf", return_value={
            "user": "zmuser", "password": "zmpass",
            "host": "localhost:/var/run/mysqld/mysqld.sock", "database": "zm",
        }), patch("mysql.connector.connect", mock_connect):
            get_zm_db()

        kw = mock_connect.call_args.kwargs
        assert kw["unix_socket"] == "/var/run/mysqld/mysqld.sock"


# ===========================================================================
# 6. TestStreamConfigVariants
# ===========================================================================

class TestStreamConfigVariants:
    """Test StreamConfig.from_dict with edge cases."""

    def test_resize_no_becomes_none(self):
        sc = StreamConfig.from_dict({"resize": "no"})
        assert sc.resize is None

    def test_resize_int(self):
        sc = StreamConfig.from_dict({"resize": 640})
        assert sc.resize == 640

    def test_resize_string_int(self):
        sc = StreamConfig.from_dict({"resize": "640"})
        assert sc.resize == 640

    def test_frame_set_comma_string(self):
        sc = StreamConfig.from_dict({"frame_set": "snapshot,alarm,1,2"})
        assert sc.frame_set == ["snapshot", "alarm", "1", "2"]

    def test_frame_set_list_mixed(self):
        sc = StreamConfig.from_dict({"frame_set": [1, "alarm"]})
        assert sc.frame_set == ["1", "alarm"]

    def test_save_frames_yes_string(self):
        sc = StreamConfig.from_dict({"save_frames": "yes"})
        assert sc.save_frames is True

    def test_save_frames_no_string(self):
        sc = StreamConfig.from_dict({"save_frames": "no"})
        assert sc.save_frames is False

    def test_unknown_keys_silently_ignored(self):
        sc = StreamConfig.from_dict({
            "api": "http://example.com",
            "polygons": [[1, 2, 3, 4]],
            "mid": 5,
            "frame_strategy": "most",
            "resize": 800,
        })
        assert sc.resize == 800
        # No error raised, and no unexpected attributes
        assert not hasattr(sc, "api")
        assert not hasattr(sc, "polygons")

    def test_convert_snapshot_to_fid_no(self):
        sc = StreamConfig.from_dict({"convert_snapshot_to_fid": "no"})
        assert sc.convert_snapshot_to_fid is False

    def test_convert_snapshot_to_fid_yes(self):
        sc = StreamConfig.from_dict({"convert_snapshot_to_fid": "yes"})
        assert sc.convert_snapshot_to_fid is True

    def test_default_values(self):
        sc = StreamConfig.from_dict({})
        assert sc.resize == 800
        assert sc.frame_set == ["snapshot", "alarm", "1"]
        assert sc.convert_snapshot_to_fid is True
        assert sc.save_frames is False

    def test_download_bool_string(self):
        sc = StreamConfig.from_dict({"download": "yes"})
        assert sc.download is True

    def test_disable_ssl_cert_check_no(self):
        sc = StreamConfig.from_dict({"disable_ssl_cert_check": "no"})
        assert sc.disable_ssl_cert_check is False

    def test_delete_after_analyze_yes(self):
        sc = StreamConfig.from_dict({"delete_after_analyze": "yes"})
        assert sc.delete_after_analyze is True


# ===========================================================================
# 7. TestGatewayConfigVariants
# ===========================================================================

class TestGatewayConfigVariants:
    """Test that Detector correctly handles various gateway configurations."""

    def test_gateway_url_mode(self):
        from pyzm.ml.detector import Detector

        det = Detector(
            config=DetectorConfig(models=[]),
            gateway="http://gpu:5000",
            gateway_mode="url",
        )
        assert det._gateway == "http://gpu:5000"
        assert det._gateway_mode == "url"

    def test_gateway_trailing_slash_stripped(self):
        from pyzm.ml.detector import Detector

        det = Detector(
            config=DetectorConfig(models=[]),
            gateway="http://gpu:5000/",
        )
        assert det._gateway == "http://gpu:5000"

    def test_gateway_none_local_mode(self):
        from pyzm.ml.detector import Detector

        det = Detector(config=DetectorConfig(models=[]))
        assert det._gateway is None

    def test_gateway_mode_image(self):
        from pyzm.ml.detector import Detector

        det = Detector(
            config=DetectorConfig(models=[]),
            gateway="http://gpu:5000",
            gateway_mode="image",
        )
        assert det._gateway_mode == "image"

    def test_gateway_timeout(self):
        from pyzm.ml.detector import Detector

        det = Detector(
            config=DetectorConfig(models=[]),
            gateway="http://gpu:5000",
            gateway_timeout=120,
        )
        assert det._gateway_timeout == 120

    def test_gateway_auth_threaded(self):
        from pyzm.ml.detector import Detector

        det = Detector(
            config=DetectorConfig(models=[]),
            gateway="http://gpu:5000",
            gateway_username="admin",
            gateway_password="secret",
        )
        assert det._gateway_username == "admin"
        assert det._gateway_password == "secret"

    def test_gateway_default_timeout(self):
        from pyzm.ml.detector import Detector

        det = Detector(
            config=DetectorConfig(models=[]),
            gateway="http://gpu:5000",
        )
        assert det._gateway_timeout == 60

    def test_from_dict_gateway_threaded(self):
        from pyzm.ml.detector import Detector

        ml = _ml_options(global_general={
            "ml_gateway": "http://gpu:5000/",
            "ml_gateway_mode": "image",
            "ml_user": "u",
            "ml_password": "p",
            "ml_timeout": 90,
        })
        det = Detector.from_dict(ml)
        assert det._gateway == "http://gpu:5000"
        assert det._gateway_mode == "image"
        assert det._gateway_username == "u"
        assert det._gateway_password == "p"
        assert det._gateway_timeout == 90
