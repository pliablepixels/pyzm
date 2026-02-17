"""Tests for pyzm.models.config -- Pydantic configuration models."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

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
    ZMClientConfig,
)


# ===================================================================
# TestZMClientConfig
# ===================================================================

class TestZMClientConfig:
    """Tests for ZMClientConfig model."""

    def test_creation_basic(self):
        cfg = ZMClientConfig(api_url="https://zm.example.com/zm/api")
        assert cfg.api_url == "https://zm.example.com/zm/api"
        assert cfg.verify_ssl is True
        assert cfg.timeout == 30
        assert cfg.user is None
        assert cfg.password is None

    def test_portal_url_auto_derived_from_api_url(self):
        cfg = ZMClientConfig(api_url="https://zm.example.com/zm/api")
        assert cfg.portal_url == "https://zm.example.com/zm"

    def test_portal_url_not_derived_when_no_api_suffix(self):
        cfg = ZMClientConfig(api_url="https://zm.example.com/zm/custom")
        assert cfg.portal_url is None

    def test_portal_url_explicit_overrides_derivation(self):
        cfg = ZMClientConfig(
            api_url="https://zm.example.com/zm/api",
            portal_url="https://custom-portal.example.com/zm",
        )
        assert cfg.portal_url == "https://custom-portal.example.com/zm"

    def test_full_creation_with_all_fields(self):
        cfg = ZMClientConfig(
            api_url="https://zm.example.com/zm/api",
            portal_url="https://zm.example.com/zm",
            user="admin",
            password="s3cret",
            basic_auth_user="web",
            basic_auth_password="basic_pw",
            verify_ssl=False,
            timeout=60,
        )
        assert cfg.user == "admin"
        assert cfg.password.get_secret_value() == "s3cret"
        assert cfg.basic_auth_user == "web"
        assert cfg.basic_auth_password.get_secret_value() == "basic_pw"
        assert cfg.verify_ssl is False
        assert cfg.timeout == 60

    def test_password_is_secret_str(self):
        cfg = ZMClientConfig(
            api_url="https://zm.example.com/zm/api",
            password="secret123",
        )
        assert isinstance(cfg.password, SecretStr)
        assert "secret123" not in repr(cfg.password)

    def test_validation_error_missing_api_url(self):
        with pytest.raises(ValidationError):
            ZMClientConfig()

    def test_validation_error_invalid_timeout_type(self):
        with pytest.raises(ValidationError):
            ZMClientConfig(api_url="https://zm.example.com/zm/api", timeout="not_int")


# ===================================================================
# TestModelConfig
# ===================================================================

class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_defaults(self):
        mc = ModelConfig()
        assert mc.name is None
        assert mc.enabled is True
        assert mc.type == ModelType.OBJECT
        assert mc.framework == ModelFramework.OPENCV
        assert mc.processor == Processor.CPU
        assert mc.min_confidence == 0.3
        assert mc.pattern == ".*"
        assert mc.model_width is None
        assert mc.model_height is None
        assert mc.pre_existing_labels == []
        assert mc.options == {}

    def test_all_field_types(self):
        mc = ModelConfig(
            name="test-model",
            enabled=False,
            type=ModelType.FACE,
            framework=ModelFramework.FACE_DLIB,
            processor=Processor.GPU,
            weights="/path/to/weights",
            config="/path/to/config",
            labels="/path/to/labels",
            min_confidence=0.7,
            pattern="(person|face)",
            max_detection_size="50%",
            model_width=640,
            model_height=640,
            known_faces_dir="/faces",
            face_model="hog",
            face_train_model="hog",
            face_recog_dist_threshold=0.5,
            face_num_jitters=2,
            face_upsample_times=2,
            alpr_service="plate_recognizer",
            alpr_key="my-key",
            alpr_url="https://api.platerecognizer.com",
            platerec_min_dscore=0.3,
            platerec_min_score=0.5,
            aws_region="eu-west-1",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="secret...",
            disable_locks=True,
            max_lock_wait=60,
            max_processes=4,
            pre_existing_labels=["person", "car"],
            options={"custom_key": "custom_value"},
        )
        assert mc.name == "test-model"
        assert mc.enabled is False
        assert mc.type == ModelType.FACE
        assert mc.framework == ModelFramework.FACE_DLIB
        assert mc.processor == Processor.GPU
        assert mc.min_confidence == 0.7
        assert mc.pre_existing_labels == ["person", "car"]
        assert mc.options == {"custom_key": "custom_value"}
        assert mc.disable_locks is True
        assert mc.max_lock_wait == 60
        assert mc.max_processes == 4

    def test_enum_validation_model_type(self):
        mc = ModelConfig(type="face")
        assert mc.type == ModelType.FACE

    def test_enum_validation_invalid_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(type="invalid_type")

    def test_enum_validation_framework(self):
        mc = ModelConfig(framework="coral_edgetpu")
        assert mc.framework == ModelFramework.CORAL

    def test_enum_validation_processor(self):
        mc = ModelConfig(processor="gpu")
        assert mc.processor == Processor.GPU


# ===================================================================
# TestDetectorConfig
# ===================================================================

class TestDetectorConfig:
    """Tests for DetectorConfig model."""

    def test_creation_defaults(self):
        dc = DetectorConfig()
        assert dc.models == []
        assert dc.match_strategy == MatchStrategy.MOST
        assert dc.frame_strategy == FrameStrategy.MOST_MODELS
        assert dc.max_detection_size is None
        assert dc.pattern == ".*"
        assert dc.match_past_detections is False
        assert dc.past_det_max_diff_area == "5%"
        assert dc.image_path == "/tmp"

    def test_creation_with_models(self):
        mc = ModelConfig(name="yolov4", type=ModelType.OBJECT)
        dc = DetectorConfig(models=[mc], match_strategy=MatchStrategy.MOST)
        assert len(dc.models) == 1
        assert dc.match_strategy == MatchStrategy.MOST

    def test_from_dict_realistic_object_sequence(self):
        """Test from_dict with a realistic ml_sequence dict as used in
        objectconfig.yml."""
        ml_options = {
            "general": {
                "model_sequence": "object,face",
                "same_model_sequence_strategy": "first",
                "frame_strategy": "most_models",
                "pattern": "(person|car|truck)",
                "disable_locks": "no",
            },
            "object": {
                "general": {
                    "pattern": "(person|car|truck)",
                    "same_model_sequence_strategy": "first",
                },
                "sequence": [
                    {
                        "name": "YOLOv4",
                        "enabled": "yes",
                        "object_framework": "opencv",
                        "object_processor": "cpu",
                        "object_weights": "/var/lib/zmeventnotification/models/yolov4/yolov4.weights",
                        "object_config": "/var/lib/zmeventnotification/models/yolov4/yolov4.cfg",
                        "object_labels": "/var/lib/zmeventnotification/models/yolov4/coco.names",
                        "object_min_confidence": "0.3",
                        "model_width": "416",
                        "model_height": "416",
                    },
                ],
            },
            "face": {
                "general": {
                    "pattern": ".*",
                    "pre_existing_labels": ["person"],
                },
                "sequence": [
                    {
                        "name": "dlib face",
                        "enabled": "yes",
                        "face_detection_framework": "dlib",
                        "known_images_path": "/var/lib/zmeventnotification/known_faces",
                        "face_model": "cnn",
                        "face_train_model": "cnn",
                        "face_recog_dist_threshold": "0.6",
                        "face_num_jitters": "1",
                        "face_upsample_times": "1",
                    },
                ],
            },
        }

        dc = DetectorConfig.from_dict(ml_options)

        # Should have 2 models: one object, one face
        assert len(dc.models) == 2

        # First model: YOLO object detection
        obj_model = dc.models[0]
        assert obj_model.name == "YOLOv4"
        assert obj_model.type == ModelType.OBJECT
        assert obj_model.framework == ModelFramework.OPENCV
        assert obj_model.processor == Processor.CPU
        assert obj_model.weights == "/var/lib/zmeventnotification/models/yolov4/yolov4.weights"
        assert obj_model.config == "/var/lib/zmeventnotification/models/yolov4/yolov4.cfg"
        assert obj_model.labels == "/var/lib/zmeventnotification/models/yolov4/coco.names"
        assert obj_model.min_confidence == 0.3
        assert obj_model.model_width == 416
        assert obj_model.model_height == 416
        assert obj_model.pattern == "(person|car|truck)"

        # Second model: face detection
        face_model = dc.models[1]
        assert face_model.name == "dlib face"
        assert face_model.type == ModelType.FACE
        assert face_model.framework == ModelFramework.FACE_DLIB
        assert face_model.known_faces_dir == "/var/lib/zmeventnotification/known_faces"
        assert face_model.face_model == "cnn"
        assert face_model.face_recog_dist_threshold == 0.6
        assert face_model.pre_existing_labels == ["person"]

        # Global settings
        assert dc.match_strategy == MatchStrategy.FIRST
        assert dc.frame_strategy == FrameStrategy.MOST_MODELS
        assert dc.pattern == "(person|car|truck)"

    def test_from_dict_disabled_model(self):
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {},
                "sequence": [
                    {
                        "name": "disabled-model",
                        "enabled": "no",
                        "object_framework": "opencv",
                        "object_min_confidence": "0.5",
                    },
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        assert len(dc.models) == 1
        assert dc.models[0].enabled is False

    def test_from_dict_unknown_model_type_skipped(self):
        ml_options = {
            "general": {"model_sequence": "object,unknown_type"},
            "object": {
                "general": {},
                "sequence": [
                    {"name": "test", "object_framework": "opencv"},
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        # unknown_type should be skipped
        assert len(dc.models) == 1

    def test_from_dict_match_past_detections(self):
        ml_options = {
            "general": {
                "model_sequence": "object",
                "match_past_detections": "yes",
                "past_det_max_diff_area": "10%",
                "image_path": "/var/cache/pyzm",
            },
            "object": {
                "general": {},
                "sequence": [
                    {"object_framework": "opencv"},
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        assert dc.match_past_detections is True
        assert dc.past_det_max_diff_area == "10%"
        assert dc.image_path == "/var/cache/pyzm"

    def test_from_dict_per_type_overrides(self):
        """Per-type section_general keys populate type_overrides."""
        ml_options = {
            "general": {
                "model_sequence": "object,face",
                "same_model_sequence_strategy": "first",
            },
            "object": {
                "general": {
                    "same_model_sequence_strategy": "most",
                    "match_past_detections": "yes",
                    "past_det_max_diff_area": "10%",
                    "car_past_det_max_diff_area": "15%",
                    "ignore_past_detection_labels": ["dog"],
                    "aliases": [["car", "bus"]],
                },
                "sequence": [{"object_framework": "opencv"}],
            },
            "face": {
                "general": {
                    "same_model_sequence_strategy": "union",
                },
                "sequence": [{"face_detection_framework": "dlib"}],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)

        # Object overrides
        assert ModelType.OBJECT in dc.type_overrides
        obj_ov = dc.type_overrides[ModelType.OBJECT]
        assert obj_ov.match_strategy == MatchStrategy.MOST
        assert obj_ov.match_past_detections is True
        assert obj_ov.past_det_max_diff_area == "10%"
        assert obj_ov.past_det_max_diff_area_labels == {"car": "15%"}
        assert obj_ov.ignore_past_detection_labels == ["dog"]
        assert obj_ov.aliases == [["car", "bus"]]

        # Face overrides
        assert ModelType.FACE in dc.type_overrides
        face_ov = dc.type_overrides[ModelType.FACE]
        assert face_ov.match_strategy == MatchStrategy.UNION
        assert face_ov.match_past_detections is None  # not set

        # Global strategy is still "first"
        assert dc.match_strategy == MatchStrategy.FIRST

    def test_from_dict_warns_on_frame_strategy_in_section(self, caplog):
        """frame_strategy in section_general should log a warning."""
        import logging
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {"frame_strategy": "first"},
                "sequence": [{"object_framework": "opencv"}],
            },
        }
        with caplog.at_level(logging.WARNING, logger="pyzm"):
            DetectorConfig.from_dict(ml_options)
        assert any("frame_strategy" in r.message and "no per-type effect" in r.message for r in caplog.records)

    def test_from_dict_warns_on_image_path_in_section(self, caplog):
        """image_path in section_general should log a warning."""
        import logging
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {"image_path": "/some/path"},
                "sequence": [{"object_framework": "opencv"}],
            },
        }
        with caplog.at_level(logging.WARNING, logger="pyzm"):
            DetectorConfig.from_dict(ml_options)
        assert any("image_path" in r.message and "no per-type effect" in r.message for r in caplog.records)

    def test_from_dict_max_detection_size_cascades(self):
        """max_detection_size in section_general cascades to ModelConfig."""
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {"max_detection_size": "30%"},
                "sequence": [
                    {"name": "no-override", "object_framework": "opencv"},
                    {"name": "has-override", "object_framework": "opencv", "max_detection_size": "10%"},
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        # First model gets section_general fallback
        assert dc.models[0].max_detection_size == "30%"
        # Second model uses its own explicit value
        assert dc.models[1].max_detection_size == "10%"

    def test_from_dict_no_type_overrides_when_only_pattern(self):
        """section_general with only non-overridable keys (like pattern) should not
        create type_overrides."""
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {"pattern": "(person|car)"},
                "sequence": [{"object_framework": "opencv"}],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        assert ModelType.OBJECT not in dc.type_overrides

    def test_type_overrides_default_empty(self):
        """Default DetectorConfig has empty type_overrides."""
        dc = DetectorConfig()
        assert dc.type_overrides == {}

    def test_from_dict_audio_birdnet_sequence(self):
        """from_dict correctly parses audio type with BirdNET config."""
        ml_options = {
            "general": {
                "model_sequence": "object,audio",
                "same_model_sequence_strategy": "first",
            },
            "object": {
                "general": {"pattern": "(person|car)"},
                "sequence": [
                    {"name": "YOLO", "object_framework": "opencv"},
                ],
            },
            "audio": {
                "general": {
                    "pattern": ".*",
                    "same_model_sequence_strategy": "first",
                },
                "sequence": [
                    {
                        "name": "BirdNET",
                        "enabled": "yes",
                        "audio_framework": "birdnet",
                        "birdnet_min_conf": "0.5",
                        "birdnet_lat": "43.65",
                        "birdnet_lon": "-79.38",
                        "birdnet_sensitivity": "1.5",
                        "birdnet_overlap": "0.5",
                    },
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)

        assert len(dc.models) == 2
        obj_model = dc.models[0]
        audio_model = dc.models[1]

        assert obj_model.type == ModelType.OBJECT
        assert audio_model.type == ModelType.AUDIO
        assert audio_model.name == "BirdNET"
        assert audio_model.framework == ModelFramework.BIRDNET
        assert audio_model.birdnet_min_conf == 0.5
        assert audio_model.birdnet_lat == 43.65
        assert audio_model.birdnet_lon == -79.38
        assert audio_model.birdnet_sensitivity == 1.5
        assert audio_model.birdnet_overlap == 0.5

    def test_from_dict_audio_defaults_framework_to_birdnet(self):
        """Audio type defaults to birdnet framework when not specified."""
        ml_options = {
            "general": {"model_sequence": "audio"},
            "audio": {
                "general": {},
                "sequence": [
                    {"name": "default-audio"},
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        assert dc.models[0].framework == ModelFramework.BIRDNET

    def test_from_dict_audio_birdnet_defaults(self):
        """BirdNET config fields have correct defaults when not specified in YAML."""
        ml_options = {
            "general": {"model_sequence": "audio"},
            "audio": {
                "general": {},
                "sequence": [{"name": "BirdNET"}],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        mc = dc.models[0]
        assert mc.birdnet_lat == -1.0
        assert mc.birdnet_lon == -1.0
        assert mc.birdnet_min_conf == 0.25
        assert mc.birdnet_sensitivity == 1.0
        assert mc.birdnet_overlap == 0.0

    def test_from_dict_audio_birdnet_min_confidence_alias(self):
        """birdnet_min_confidence works as alias for birdnet_min_conf."""
        ml_options = {
            "general": {"model_sequence": "audio"},
            "audio": {
                "general": {},
                "sequence": [
                    {
                        "name": "BirdNET",
                        "birdnet_min_confidence": "0.7",
                    },
                ],
            },
        }
        dc = DetectorConfig.from_dict(ml_options)
        assert dc.models[0].birdnet_min_conf == 0.7


# ===================================================================
# TestModelConfigBirdnet
# ===================================================================

class TestModelConfigBirdnet:
    """Tests for BirdNET-specific ModelConfig fields."""

    def test_birdnet_field_defaults(self):
        mc = ModelConfig()
        assert mc.birdnet_lat == -1.0
        assert mc.birdnet_lon == -1.0
        assert mc.birdnet_min_conf == 0.25
        assert mc.birdnet_sensitivity == 1.0
        assert mc.birdnet_overlap == 0.0

    def test_birdnet_fields_set(self):
        mc = ModelConfig(
            type=ModelType.AUDIO,
            framework=ModelFramework.BIRDNET,
            birdnet_lat=51.5,
            birdnet_lon=-0.1,
            birdnet_min_conf=0.7,
            birdnet_sensitivity=1.5,
            birdnet_overlap=0.5,
        )
        assert mc.type == ModelType.AUDIO
        assert mc.framework == ModelFramework.BIRDNET
        assert mc.birdnet_lat == 51.5
        assert mc.birdnet_lon == -0.1
        assert mc.birdnet_min_conf == 0.7
        assert mc.birdnet_sensitivity == 1.5
        assert mc.birdnet_overlap == 0.5

    def test_audio_enum_values(self):
        assert ModelType.AUDIO.value == "audio"
        assert ModelFramework.BIRDNET.value == "birdnet"

    def test_audio_enum_from_string(self):
        mc = ModelConfig(type="audio", framework="birdnet")
        assert mc.type == ModelType.AUDIO
        assert mc.framework == ModelFramework.BIRDNET


# ===================================================================
# TestStreamConfig
# ===================================================================

class TestStreamConfig:
    """Tests for StreamConfig model."""

    def test_defaults(self):
        sc = StreamConfig()
        assert sc.frame_set == ["snapshot", "alarm", "1"]
        assert sc.max_frames == 0
        assert sc.resize == 800
        assert sc.start_frame == 1
        assert sc.frame_skip == 1
        assert sc.download is False
        assert sc.download_dir == "/tmp"
        assert sc.delay == 0
        assert sc.delay_between_frames == 0
        assert sc.delay_between_snapshots == 0
        assert sc.contig_frames_before_error == 5
        assert sc.max_attempts == 1
        assert sc.sleep_between_attempts == 3
        assert sc.disable_ssl_cert_check is True
        assert sc.save_frames is False
        assert sc.save_frames_dir == "/tmp"
        assert sc.delete_after_analyze is False
        assert sc.convert_snapshot_to_fid is False

    def test_all_fields_override(self):
        sc = StreamConfig(
            frame_set=["1", "5", "10"],
            max_frames=3,
            resize=640,
            start_frame=2,
            frame_skip=2,
            download=True,
            download_dir="/data/downloads",
            delay=5,
            delay_between_frames=1,
            delay_between_snapshots=2,
            contig_frames_before_error=10,
            max_attempts=3,
            sleep_between_attempts=5,
            disable_ssl_cert_check=False,
            save_frames=True,
            save_frames_dir="/data/frames",
            delete_after_analyze=True,
            convert_snapshot_to_fid=True,
        )
        assert sc.frame_set == ["1", "5", "10"]
        assert sc.max_frames == 3
        assert sc.resize == 640
        assert sc.start_frame == 2
        assert sc.frame_skip == 2
        assert sc.download is True
        assert sc.download_dir == "/data/downloads"
        assert sc.delay == 5
        assert sc.delay_between_frames == 1
        assert sc.delay_between_snapshots == 2
        assert sc.contig_frames_before_error == 10
        assert sc.max_attempts == 3
        assert sc.sleep_between_attempts == 5
        assert sc.disable_ssl_cert_check is False
        assert sc.save_frames is True
        assert sc.save_frames_dir == "/data/frames"
        assert sc.delete_after_analyze is True
        assert sc.convert_snapshot_to_fid is True

    def test_resize_none_disables_resizing(self):
        sc = StreamConfig(resize=None)
        assert sc.resize is None

    def test_empty_frame_set(self):
        sc = StreamConfig(frame_set=[])
        assert sc.frame_set == []


class TestStreamConfigFromDict:
    """Tests for StreamConfig.from_dict() legacy dict conversion."""

    def test_basic_conversion(self):
        d = {
            "resize": "800",
            "frame_set": "snapshot,alarm,1",
            "contig_frames_before_error": "5",
            "max_attempts": "3",
            "sleep_between_attempts": "4",
        }
        sc = StreamConfig.from_dict(d)
        assert sc.resize == 800
        assert sc.frame_set == ["snapshot", "alarm", "1"]
        assert sc.contig_frames_before_error == 5
        assert sc.max_attempts == 3
        assert sc.sleep_between_attempts == 4

    def test_resize_no_means_none(self):
        sc = StreamConfig.from_dict({"resize": "no"})
        assert sc.resize is None

    def test_resize_none_means_none(self):
        sc = StreamConfig.from_dict({"resize": None})
        assert sc.resize is None

    def test_resize_numeric_int(self):
        sc = StreamConfig.from_dict({"resize": 640})
        assert sc.resize == 640

    def test_frame_set_as_list(self):
        sc = StreamConfig.from_dict({"frame_set": [1, 5, "alarm"]})
        assert sc.frame_set == ["1", "5", "alarm"]

    def test_boolean_yes_no_conversion(self):
        sc = StreamConfig.from_dict({
            "download": "yes",
            "save_frames": "no",
            "delete_after_analyze": "yes",
            "convert_snapshot_to_fid": "no",
        })
        assert sc.download is True
        assert sc.save_frames is False
        assert sc.delete_after_analyze is True
        assert sc.convert_snapshot_to_fid is False

    def test_ignores_unknown_keys(self):
        """Keys like 'api', 'polygons', 'mid' should be silently ignored."""
        sc = StreamConfig.from_dict({
            "api": object(),
            "polygons": [],
            "mid": "5",
            "frame_strategy": "most_models",
            "resize": "800",
        })
        assert sc.resize == 800

    def test_empty_dict_gives_defaults(self):
        sc = StreamConfig.from_dict({})
        default = StreamConfig()
        assert sc.frame_set == default.frame_set
        assert sc.resize == default.resize

    def test_realistic_objectconfig(self):
        """Realistic stream_sequence from objectconfig.yml."""
        d = {
            "frame_strategy": "most_models",
            "frame_set": "snapshot,alarm",
            "contig_frames_before_error": 5,
            "max_attempts": 3,
            "sleep_between_attempts": 4,
            "resize": 800,
        }
        sc = StreamConfig.from_dict(d)
        assert sc.frame_set == ["snapshot", "alarm"]
        assert sc.resize == 800
        assert sc.max_attempts == 3
