"""Tests for pyzm.ml.detector -- top-level Detector API."""

from __future__ import annotations

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
)
from pyzm.models.detection import BBox, Detection, DetectionResult


# ===================================================================
# Helpers
# ===================================================================

def _det(label: str, conf: float = 0.9) -> Detection:
    return Detection(
        label=label,
        confidence=conf,
        bbox=BBox(x1=10, y1=10, x2=50, y2=50),
        model_name="test",
    )


def _make_pipeline_mock(detections: list[Detection] | None = None) -> MagicMock:
    """Create a mock ModelPipeline."""
    pipeline = MagicMock()
    result = DetectionResult(detections=detections or [])
    pipeline.run.return_value = result
    return pipeline


# ===================================================================
# TestDetectorInit
# ===================================================================

class TestDetectorInit:
    """Tests for Detector.__init__ with various argument types."""

    def test_init_with_model_name_strings(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])
        assert len(det._config.models) == 1
        assert det._config.models[0].name == "yolov4"

    def test_init_with_multiple_model_names(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4", "myface"])
        assert len(det._config.models) == 2
        assert det._config.models[0].name == "yolov4"
        assert det._config.models[1].name == "myface"

    def test_init_with_model_config_objects(self):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(
            name="custom-model",
            type=ModelType.OBJECT,
            framework=ModelFramework.OPENCV,
            weights="/path/to/weights",
        )
        det = Detector(models=[mc])
        assert len(det._config.models) == 1
        assert det._config.models[0] is mc

    def test_init_with_mixed_strings_and_configs(self):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(name="custom", type=ModelType.FACE)
        det = Detector(models=["yolov4", mc])
        assert len(det._config.models) == 2
        assert det._config.models[0].name == "yolov4"
        assert det._config.models[1].name == "custom"

    def test_init_with_detector_config(self):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(name="yolov7", framework=ModelFramework.OPENCV)
        config = DetectorConfig(
            models=[mc],
            match_strategy=MatchStrategy.MOST,
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)
        assert det._config is config
        assert det._config.match_strategy == MatchStrategy.MOST

    def test_init_with_no_args_empty_base(self):
        from pyzm.ml.detector import Detector

        det = Detector(base_path="/nonexistent/path")
        assert det._config.models == []

    def test_init_unknown_model_creates_bare_config(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["nonexistent_model"])
        mc = det._config.models[0]
        assert mc.name == "nonexistent_model"
        assert mc.weights is None


# ===================================================================
# TestDetectorDetect
# ===================================================================

class TestDetectorDetect:
    """Tests for Detector.detect() method."""

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_single_image_mock(self, mock_pipeline_cls):
        """Test detect() with a single numpy array (mock).

        The detect() method does ``import numpy as np`` lazily and then
        ``isinstance(input, np.ndarray)``.  We mock numpy at the
        sys.modules level so the isinstance check works with our mock
        image.
        """
        pipeline = _make_pipeline_mock([_det("person"), _det("car")])
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])

        # Create a proper mock numpy module with an ndarray class
        import types
        mock_np_module = types.ModuleType("numpy")
        mock_image = MagicMock()

        # Make isinstance(mock_image, mock_np_module.ndarray) return True
        class FakeNdarray:
            pass

        mock_np_module.ndarray = FakeNdarray  # type: ignore[attr-defined]
        mock_image.__class__ = FakeNdarray
        mock_image.shape = (100, 100, 3)

        with patch.dict("sys.modules", {"numpy": mock_np_module}):
            result = det.detect(mock_image)

        assert isinstance(result, DetectionResult)
        assert result.frame_id == "single"

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_multiple_frames_most_strategy(self, mock_pipeline_cls):
        """Test detect() with multiple frames and MOST frame strategy."""
        # Frame 1: 1 detection, Frame 2: 3 detections
        result1 = DetectionResult(detections=[_det("person")])
        result2 = DetectionResult(detections=[_det("person"), _det("car"), _det("dog")])

        pipeline = MagicMock()
        pipeline.run.side_effect = [result1, result2]
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        config = DetectorConfig(
            models=[ModelConfig(name="test")],
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)

        # Create mock frames
        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        frames = [(1, mock_img1), (2, mock_img2)]

        result = det.detect(frames)

        # Should pick frame 2 (3 detections > 1)
        assert len(result.detections) == 3
        assert result.frame_id == 2

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_string_path(self, mock_pipeline_cls):
        """Test detect() with a file path."""
        pipeline = _make_pipeline_mock([_det("person")])
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])

        # Mock cv2.imread
        mock_cv2 = MagicMock()
        mock_image = MagicMock()
        mock_cv2.imread.return_value = mock_image

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = det.detect("/path/to/image.jpg")

        assert result.frame_id == "single"

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_string_path_not_found(self, mock_pipeline_cls):
        """Test detect() raises FileNotFoundError for bad path."""
        mock_pipeline_cls.return_value = MagicMock()

        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])

        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = None  # cv2 returns None for bad files

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with pytest.raises(FileNotFoundError):
                det.detect("/nonexistent/image.jpg")

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_multiple_frames_first_strategy(self, mock_pipeline_cls):
        """Test FIRST frame strategy: returns first matched frame."""
        result1 = DetectionResult(detections=[_det("person")])
        result2 = DetectionResult(detections=[_det("person"), _det("car")])

        pipeline = MagicMock()
        pipeline.run.side_effect = [result1, result2]
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        config = DetectorConfig(
            models=[ModelConfig(name="test")],
            frame_strategy=FrameStrategy.FIRST,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock()), (2, MagicMock())]
        result = det.detect(frames)

        # Should return first matched frame
        assert result.frame_id == 1
        assert len(result.detections) == 1


# ===================================================================
# TestModelDiscovery
# ===================================================================

class TestModelDiscovery:
    """Tests for base_path auto-discovery and name resolution."""

    def _make_darknet_dir(self, base, name, weights_name=None, cfg_name=None, labels_name="coco.names"):
        """Create a directory with Darknet-style model files."""
        d = base / name
        d.mkdir()
        (d / (weights_name or f"{name}.weights")).touch()
        if cfg_name is not False:
            (d / (cfg_name or f"{name}.cfg")).touch()
        (d / labels_name).touch()
        return d

    def _make_onnx_dir(self, base, dirname, *model_names):
        """Create a directory with ONNX model files."""
        d = base / dirname
        d.mkdir()
        for n in model_names:
            (d / f"{n}.onnx").touch()
        return d

    def test_discover_darknet_models(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        self._make_darknet_dir(tmp_path, "yolov4")
        models = _discover_models(tmp_path)

        assert len(models) == 1
        assert models[0].name == "yolov4"
        assert models[0].weights.endswith("yolov4.weights")
        assert models[0].config.endswith("yolov4.cfg")
        assert models[0].labels.endswith("coco.names")
        assert models[0].framework == ModelFramework.OPENCV

    def test_discover_onnx_models(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26n", "yolo26s")
        models = _discover_models(tmp_path)

        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"yolo26n", "yolo26s"}
        for m in models:
            assert m.weights.endswith(".onnx")
            assert m.config is None
            assert m.labels is None

    def test_discover_tflite_models(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        d = tmp_path / "coral"
        d.mkdir()
        (d / "ssd_mobilenet.tflite").touch()
        (d / "coco_labels.txt").touch()

        models = _discover_models(d.parent)
        coral_models = [m for m in models if m.name == "ssd_mobilenet"]
        assert len(coral_models) == 1
        assert coral_models[0].framework == ModelFramework.CORAL
        assert coral_models[0].processor == Processor.TPU
        assert coral_models[0].labels.endswith("coco_labels.txt")

    def test_discover_mixed_dirs(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        self._make_darknet_dir(tmp_path, "yolov4")
        self._make_onnx_dir(tmp_path, "ultralytics", "yolo11n", "yolo26s")

        models = _discover_models(tmp_path)
        assert len(models) == 3
        names = {m.name for m in models}
        assert "yolov4" in names
        assert "yolo11n" in names
        assert "yolo26s" in names

    def test_discover_empty_dir(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        models = _discover_models(tmp_path)
        assert models == []

    def test_discover_nonexistent_dir(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        models = _discover_models(tmp_path / "nonexistent")
        assert models == []

    def test_resolve_by_directory_name(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        self._make_darknet_dir(tmp_path, "yolov4")
        mc = _resolve_model_name("yolov4", tmp_path)

        assert mc.name == "yolov4"
        assert mc.weights.endswith("yolov4.weights")
        assert mc.config.endswith("yolov4.cfg")

    def test_resolve_by_file_stem(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26s", "yolo26n")
        mc = _resolve_model_name("yolo26s", tmp_path)

        assert mc.name == "yolo26s"
        assert mc.weights.endswith("yolo26s.onnx")

    def test_resolve_falls_back_to_bare_config(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        mc = _resolve_model_name("unknown_model", tmp_path)
        assert mc.name == "unknown_model"
        assert mc.weights is None  # no files found
        assert mc.processor == Processor.CPU  # default

    def test_resolve_fallback_uses_provided_processor(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        mc = _resolve_model_name("unknown_model", tmp_path, Processor.GPU)
        assert mc.name == "unknown_model"
        assert mc.processor == Processor.GPU

    def test_detector_init_with_base_path(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        det = Detector(models=["yolov4"], base_path=tmp_path)

        assert len(det._config.models) == 1
        assert det._config.models[0].weights.endswith("yolov4.weights")

    def test_detector_init_auto_discover(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26s")

        det = Detector(base_path=tmp_path)
        assert len(det._config.models) == 2

    def test_detector_init_auto_discover_empty(self, tmp_path):
        from pyzm.ml.detector import Detector

        det = Detector(base_path=tmp_path)
        assert det._config.models == []

    def test_processor_string_gpu(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        det = Detector(models=["yolov4"], base_path=tmp_path, processor="gpu")

        assert det._config.models[0].processor == Processor.GPU

    def test_processor_enum_gpu(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        det = Detector(models=["yolov4"], base_path=tmp_path, processor=Processor.GPU)

        assert det._config.models[0].processor == Processor.GPU

    def test_processor_applies_to_auto_discover(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26s")

        det = Detector(base_path=tmp_path, processor="gpu")
        for m in det._config.models:
            assert m.processor == Processor.GPU

    def test_processor_does_not_override_tflite(self, tmp_path):
        from pyzm.ml.detector import Detector

        d = tmp_path / "coral"
        d.mkdir()
        (d / "model.tflite").touch()
        (d / "labels.txt").touch()

        det = Detector(base_path=tmp_path, processor="gpu")
        tflite_models = [m for m in det._config.models if m.name == "model"]
        assert tflite_models[0].processor == Processor.TPU

    def test_processor_not_applied_to_model_config_objects(self, tmp_path):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(name="custom", type=ModelType.OBJECT, processor=Processor.CPU)
        det = Detector(models=[mc], base_path=tmp_path, processor="gpu")

        assert det._config.models[0].processor == Processor.CPU


# ===================================================================
# TestDetectorFromDict
# ===================================================================

class TestDetectorFromDict:
    """Tests for Detector.from_dict() class method."""

    def test_from_dict(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object",
                "same_model_sequence_strategy": "first",
                "frame_strategy": "most_models",
            },
            "object": {
                "general": {"pattern": "(person|car)"},
                "sequence": [
                    {
                        "name": "YOLOv4",
                        "object_framework": "opencv",
                        "object_processor": "cpu",
                        "object_weights": "/models/yolov4.weights",
                        "object_config": "/models/yolov4.cfg",
                        "object_labels": "/models/coco.names",
                        "object_min_confidence": "0.3",
                    }
                ],
            },
        }

        det = Detector.from_dict(ml_options)
        assert len(det._config.models) == 1
        assert det._config.models[0].name == "YOLOv4"
        assert det._config.match_strategy == MatchStrategy.FIRST
        assert det._config.frame_strategy == FrameStrategy.MOST_MODELS

    def test_from_dict_multi_model(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object,face",
                "same_model_sequence_strategy": "first",
            },
            "object": {
                "general": {},
                "sequence": [
                    {"name": "YOLO", "object_framework": "opencv"},
                ],
            },
            "face": {
                "general": {"pre_existing_labels": ["person"]},
                "sequence": [
                    {"name": "dlib", "face_detection_framework": "dlib"},
                ],
            },
        }

        det = Detector.from_dict(ml_options)
        assert len(det._config.models) == 2
        assert det._config.models[0].type == ModelType.OBJECT
        assert det._config.models[1].type == ModelType.FACE


# ===================================================================
# TestIsBetter
# ===================================================================

class TestIsBetter:
    """Tests for the _is_better helper function."""

    def test_most_strategy(self):
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person"), _det("car")])
        r2 = DetectionResult(detections=[_det("person")])
        assert _is_better(r1, r2, FrameStrategy.MOST) is True
        assert _is_better(r2, r1, FrameStrategy.MOST) is False

    def test_most_unique_strategy(self):
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person"), _det("car")])
        r2 = DetectionResult(detections=[_det("person"), _det("person")])
        assert _is_better(r1, r2, FrameStrategy.MOST_UNIQUE) is True

    def test_most_models_strategy(self):
        from pyzm.ml.detector import _is_better

        d1 = Detection(label="person", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="model_a")
        d2 = Detection(label="car", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="model_b")
        d3 = Detection(label="person", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="model_a")

        r1 = DetectionResult(detections=[d1, d2])  # 2 models
        r2 = DetectionResult(detections=[d3])       # 1 model
        assert _is_better(r1, r2, FrameStrategy.MOST_MODELS) is True

    def test_first_strategy(self):
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person")])
        r2 = DetectionResult(detections=[])
        assert _is_better(r1, r2, FrameStrategy.FIRST) is True
        assert _is_better(r2, r1, FrameStrategy.FIRST) is False

    # -- Confidence tiebreak tests (Ref: ZoneMinder/pyzm#36) --

    def test_most_strategy_confidence_tiebreak(self):
        """When detection counts are equal, higher total confidence wins."""
        from pyzm.ml.detector import _is_better

        r_high = DetectionResult(detections=[_det("person", 0.95), _det("car", 0.90)])
        r_low = DetectionResult(detections=[_det("person", 0.60), _det("car", 0.50)])
        assert _is_better(r_high, r_low, FrameStrategy.MOST) is True
        assert _is_better(r_low, r_high, FrameStrategy.MOST) is False

    def test_most_unique_confidence_tiebreak(self):
        """When unique label counts are equal, higher total confidence wins."""
        from pyzm.ml.detector import _is_better

        r_high = DetectionResult(detections=[_det("person", 0.95), _det("car", 0.90)])
        r_low = DetectionResult(detections=[_det("person", 0.60), _det("car", 0.50)])
        assert _is_better(r_high, r_low, FrameStrategy.MOST_UNIQUE) is True
        assert _is_better(r_low, r_high, FrameStrategy.MOST_UNIQUE) is False

    def test_most_models_confidence_tiebreak(self):
        """When model count and detection count are equal, confidence breaks tie."""
        from pyzm.ml.detector import _is_better

        d1a = Detection(label="person", confidence=0.95, bbox=BBox(0, 0, 10, 10), model_name="m1")
        d1b = Detection(label="car", confidence=0.90, bbox=BBox(0, 0, 10, 10), model_name="m2")
        d2a = Detection(label="person", confidence=0.50, bbox=BBox(0, 0, 10, 10), model_name="m1")
        d2b = Detection(label="car", confidence=0.40, bbox=BBox(0, 0, 10, 10), model_name="m2")

        r_high = DetectionResult(detections=[d1a, d1b])
        r_low = DetectionResult(detections=[d2a, d2b])
        assert _is_better(r_high, r_low, FrameStrategy.MOST_MODELS) is True
        assert _is_better(r_low, r_high, FrameStrategy.MOST_MODELS) is False

    def test_most_strategy_equal_confidence_no_change(self):
        """When everything is equal, candidate does not beat current."""
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person", 0.90)])
        r2 = DetectionResult(detections=[_det("person", 0.90)])
        assert _is_better(r1, r2, FrameStrategy.MOST) is False

    # -- FIRST_NEW strategy tests (Ref: ZoneMinder/pyzm#35) --

    def test_first_new_strategy(self):
        """FIRST_NEW behaves like FIRST in _is_better (pipeline handles past-detection)."""
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person")])
        r2 = DetectionResult(detections=[])
        assert _is_better(r1, r2, FrameStrategy.FIRST_NEW) is True
        assert _is_better(r2, r1, FrameStrategy.FIRST_NEW) is False


# ===================================================================
# TestDetectorRemoteMode
# ===================================================================

class TestDetectorRemoteMode:
    """Tests for Detector gateway / remote mode."""

    def test_init_with_gateway(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"], gateway="http://gpu:5000")
        assert det._gateway == "http://gpu:5000"
        assert det._gateway_timeout == 60

    def test_init_gateway_trailing_slash_stripped(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"], gateway="http://gpu:5000/")
        assert det._gateway == "http://gpu:5000"

    def test_init_gateway_none_by_default(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])
        assert det._gateway is None

    def test_from_dict_picks_up_ml_gateway(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object",
                "same_model_sequence_strategy": "first",
                "ml_gateway": "http://gpu:5000",
                "ml_user": "admin",
                "ml_password": "secret",
                "ml_timeout": "30",
            },
            "object": {
                "general": {},
                "sequence": [{"name": "yolov4", "object_framework": "opencv"}],
            },
        }
        det = Detector.from_dict(ml_options)
        assert det._gateway == "http://gpu:5000"
        assert det._gateway_username == "admin"
        assert det._gateway_password == "secret"
        assert det._gateway_timeout == 30

    def test_from_dict_no_gateway(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {"model_sequence": "object", "same_model_sequence_strategy": "first"},
            "object": {
                "general": {},
                "sequence": [{"name": "yolov4", "object_framework": "opencv"}],
            },
        }
        det = Detector.from_dict(ml_options)
        assert det._gateway is None

    @patch("pyzm.ml.detector.Detector._remote_detect")
    def test_detect_routes_to_remote(self, mock_remote):
        """When gateway is set, detect() calls _remote_detect."""
        from pyzm.ml.detector import Detector

        mock_remote.return_value = DetectionResult(detections=[_det("person")])
        det = Detector(models=["yolov4"], gateway="http://gpu:5000")

        import types
        mock_np = types.ModuleType("numpy")

        class FakeNdarray:
            pass

        mock_np.ndarray = FakeNdarray
        image = MagicMock()
        image.__class__ = FakeNdarray

        with patch.dict("sys.modules", {"numpy": mock_np}):
            result = det.detect(image)

        mock_remote.assert_called_once()
        assert result.frame_id == "single"

    @patch("pyzm.ml.detector.Detector._remote_detect")
    def test_detect_string_routes_to_remote(self, mock_remote):
        """When gateway is set, detect(path) loads image locally then remotes."""
        from pyzm.ml.detector import Detector

        mock_remote.return_value = DetectionResult(detections=[_det("car")])
        det = Detector(models=["yolov4"], gateway="http://gpu:5000")

        mock_cv2 = MagicMock()
        mock_image = MagicMock()
        mock_cv2.imread.return_value = mock_image

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = det.detect("/path/to/image.jpg")

        mock_remote.assert_called_once()
        assert result.labels == ["car"]

    def test_init_gateway_mode_default(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"], gateway="http://gpu:5000")
        assert det._gateway_mode == "url"

    def test_init_gateway_mode_image(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"], gateway="http://gpu:5000", gateway_mode="image")
        assert det._gateway_mode == "image"

    def test_from_dict_picks_up_ml_gateway_mode(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object",
                "same_model_sequence_strategy": "first",
                "ml_gateway": "http://gpu:5000",
                "ml_gateway_mode": "url",
            },
            "object": {
                "general": {},
                "sequence": [{"name": "yolov4", "object_framework": "opencv"}],
            },
        }
        det = Detector.from_dict(ml_options)
        assert det._gateway_mode == "url"

    def test_from_dict_gateway_mode_defaults_to_url(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object",
                "same_model_sequence_strategy": "first",
                "ml_gateway": "http://gpu:5000",
            },
            "object": {
                "general": {},
                "sequence": [{"name": "yolov4", "object_framework": "opencv"}],
            },
        }
        det = Detector.from_dict(ml_options)
        assert det._gateway_mode == "url"

    @patch("pyzm.ml.detector.Detector._remote_detect_urls")
    def test_detect_event_url_mode(self, mock_remote_urls):
        """URL-mode detect_event sends frame URLs instead of fetching frames."""
        from pyzm.ml.detector import Detector

        mock_remote_urls.return_value = DetectionResult(detections=[_det("person")])
        det = Detector(models=["yolov4"], gateway="http://gpu:5000", gateway_mode="url")

        # Mock zm_client with api.portal_url and api.auth
        mock_zm = MagicMock()
        mock_zm.api.portal_url = "https://zm.example.com/zm"
        mock_zm.api.auth.get_auth_string.return_value = "token=abc123"
        mock_zm.api.config.verify_ssl = False

        from pyzm.models.config import StreamConfig
        sc = StreamConfig(frame_set=["snapshot", "alarm"])

        result = det.detect_event(mock_zm, 12345, stream_config=sc)

        mock_remote_urls.assert_called_once()
        call_args = mock_remote_urls.call_args
        frame_urls = call_args[0][0]
        assert len(frame_urls) == 2
        assert frame_urls[0]["frame_id"] == "snapshot"
        assert "eid=12345" in frame_urls[0]["url"]
        assert "fid=snapshot" in frame_urls[0]["url"]
        assert frame_urls[1]["frame_id"] == "alarm"
        assert call_args[0][1] == "token=abc123"  # zm_auth
        assert call_args[0][3] is False  # verify_ssl
        assert result.labels == ["person"]

    @patch("pyzm.ml.detector.Detector._remote_detect_urls")
    def test_detect_event_url_mode_default_frame_set(self, mock_remote_urls):
        """URL-mode with empty frame_set defaults to ['snapshot']."""
        from pyzm.ml.detector import Detector

        mock_remote_urls.return_value = DetectionResult()
        det = Detector(models=["yolov4"], gateway="http://gpu:5000", gateway_mode="url")

        mock_zm = MagicMock()
        mock_zm.api.portal_url = "https://zm.example.com/zm"
        mock_zm.api.auth.get_auth_string.return_value = ""
        mock_zm.api.config.verify_ssl = True

        from pyzm.models.config import StreamConfig
        sc = StreamConfig(frame_set=[])

        det.detect_event(mock_zm, 999, stream_config=sc)
        frame_urls = mock_remote_urls.call_args[0][0]
        assert len(frame_urls) == 1
        assert frame_urls[0]["frame_id"] == "snapshot"

    def test_detect_event_url_mode_requires_api(self):
        """URL-mode raises AttributeError if zm_client has no .api."""
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"], gateway="http://gpu:5000", gateway_mode="url")

        mock_zm = MagicMock(spec=[])  # no attributes
        with pytest.raises(AttributeError):
            det.detect_event(mock_zm, 12345)

    @patch("requests.post")
    def test_remote_detect_urls_sends_post(self, mock_post):
        """_remote_detect_urls POSTs JSON to /detect_urls."""
        from pyzm.ml.detector import Detector

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "labels": ["car"], "boxes": [[10, 20, 50, 80]],
            "confidences": [0.9], "model_names": ["yolov4"],
        }
        mock_post.return_value = mock_resp

        det = Detector(models=["yolov4"], gateway="http://gpu:5000", gateway_mode="url")

        frame_urls = [{"frame_id": "snapshot", "url": "http://zm/image?eid=1&fid=snapshot"}]
        result = det._remote_detect_urls(frame_urls, "token=abc", verify_ssl=False)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "http://gpu:5000/detect_urls"
        payload = call_kwargs[1]["json"]
        assert payload["urls"] == frame_urls
        assert payload["zm_auth"] == "token=abc"
        assert payload["verify_ssl"] is False
        assert result.labels == ["car"]

    @patch("requests.post")
    def test_remote_detect_urls_with_auth(self, mock_post):
        """_remote_detect_urls includes Bearer token when gateway auth is set."""
        from pyzm.ml.detector import Detector

        # Mock login
        mock_login_resp = MagicMock()
        mock_login_resp.json.return_value = {"access_token": "jwt123"}
        # Mock detect_urls
        mock_detect_resp = MagicMock()
        mock_detect_resp.json.return_value = {"labels": [], "boxes": [], "confidences": []}
        mock_post.side_effect = [mock_login_resp, mock_detect_resp]

        det = Detector(
            models=["yolov4"], gateway="http://gpu:5000", gateway_mode="url",
            gateway_username="admin", gateway_password="secret",
        )

        det._remote_detect_urls([{"frame_id": "1", "url": "http://zm/img"}], "token=x")

        # Second call is detect_urls
        detect_call = mock_post.call_args_list[1]
        assert detect_call[1]["headers"]["Authorization"] == "Bearer jwt123"


# ===================================================================
# TestSessionLocking (Ref: ZoneMinder/pyzm#43)
# ===================================================================

class TestSessionLocking:
    """Tests for session-level locking in _detect_multi_frame."""

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_session_lock_acquired_and_released(self, mock_pipeline_cls):
        """Exclusive backends get locked before frames and released after."""
        from pyzm.ml.detector import Detector

        # Create mock backend that needs exclusive lock
        mock_backend = MagicMock()
        mock_backend.needs_exclusive_lock = True

        # Setup pipeline with the locked backend
        pipeline = MagicMock()
        pipeline._backends = [(MagicMock(), mock_backend)]
        result = DetectionResult(detections=[_det("person")])
        pipeline.run.return_value = result
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="coral_model")],
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock()), (2, MagicMock())]
        det.detect(frames)

        mock_backend.acquire_lock.assert_called_once()
        mock_backend.release_lock.assert_called_once()

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_session_lock_released_on_exception(self, mock_pipeline_cls):
        """Lock is released even if detection raises an exception."""
        from pyzm.ml.detector import Detector

        mock_backend = MagicMock()
        mock_backend.needs_exclusive_lock = True

        pipeline = MagicMock()
        pipeline._backends = [(MagicMock(), mock_backend)]
        pipeline.run.side_effect = RuntimeError("GPU error")
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="coral_model")],
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock())]
        # Should not raise — error is caught per-frame
        det.detect(frames)

        mock_backend.acquire_lock.assert_called_once()
        mock_backend.release_lock.assert_called_once()

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_non_exclusive_backend_not_locked(self, mock_pipeline_cls):
        """Backends without needs_exclusive_lock are not locked."""
        from pyzm.ml.detector import Detector

        mock_backend = MagicMock()
        mock_backend.needs_exclusive_lock = False

        pipeline = MagicMock()
        pipeline._backends = [(MagicMock(), mock_backend)]
        result = DetectionResult(detections=[_det("person")])
        pipeline.run.return_value = result
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="yolo")],
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock())]
        det.detect(frames)

        mock_backend.acquire_lock.assert_not_called()
        mock_backend.release_lock.assert_not_called()


# ===================================================================
# TestFirstNewMultiFrame (Ref: ZoneMinder/pyzm#35)
# ===================================================================

class TestFirstNewMultiFrame:
    """Tests for FIRST_NEW frame strategy in multi-frame detection."""

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_first_new_short_circuits_on_match(self, mock_pipeline_cls):
        """FIRST_NEW returns as soon as a frame has detections."""
        from pyzm.ml.detector import Detector

        result1 = DetectionResult(detections=[_det("person")])
        result2 = DetectionResult(detections=[_det("person"), _det("car")])

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.side_effect = [result1, result2]
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="test")],
            frame_strategy=FrameStrategy.FIRST_NEW,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock()), (2, MagicMock())]
        result = det.detect(frames)

        # Should short-circuit on frame 1
        assert result.frame_id == 1
        assert len(result.detections) == 1
        # Pipeline.run should only be called once (short-circuit)
        assert pipeline.run.call_count == 1

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_first_new_skips_empty_frames(self, mock_pipeline_cls):
        """FIRST_NEW skips frames with no detections."""
        from pyzm.ml.detector import Detector

        result1 = DetectionResult(detections=[])
        result2 = DetectionResult(detections=[_det("dog")])

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.side_effect = [result1, result2]
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="test")],
            frame_strategy=FrameStrategy.FIRST_NEW,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock()), (2, MagicMock())]
        result = det.detect(frames)

        assert result.frame_id == 2
        assert result.detections[0].label == "dog"


# ===================================================================
# TestExtractEventAudio
# ===================================================================

class TestExtractEventAudio:
    """Tests for Detector._extract_event_audio() static method.

    The method uses get_zm_db() from pyzm.zm.db for direct DB access
    (not zm_client.db) and zm_client.event(eid).path() for file paths.
    """

    @staticmethod
    def _make_mock_db(row=None, raise_on_execute=False):
        """Create a mock DB connection with cursor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        if raise_on_execute:
            mock_cursor.execute.side_effect = RuntimeError("DB error")
        else:
            mock_cursor.fetchone.return_value = row
        return mock_conn

    def test_returns_none_when_no_db(self):
        """zm_client._get_db() returning None means no audio extraction."""
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = None
        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None
        assert week == -1
        assert lat == -1.0
        assert lon == -1.0

    def test_returns_none_when_db_query_fails(self):
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(raise_on_execute=True)

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None

    def test_returns_none_when_no_default_video(self):
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(
            row={"DefaultVideo": None, "StartDateTime": None, "Latitude": None, "Longitude": None},
        )

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None

    def test_returns_none_when_get_db_raises(self):
        """_get_db() raising an exception is handled gracefully."""
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.side_effect = ImportError("no mysql")

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None

    @patch("os.path.isfile", return_value=False)
    def test_returns_none_when_video_file_missing(self, mock_isfile):
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-06-15 10:00:00",
            "Latitude": None,
            "Longitude": None,
        })
        mock_zm.event.return_value.path.return_value = "/events/123"

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None

    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_returns_none_when_no_audio_stream(self, mock_isfile, mock_run):
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-06-15 10:00:00",
            "Latitude": None,
            "Longitude": None,
        })
        mock_zm.event.return_value.path.return_value = "/events/123"

        # ffprobe returns no audio
        mock_run.return_value = MagicMock(stdout="", stderr="")

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None

    @patch("os.close")
    @patch("tempfile.mkstemp", return_value=(5, "/tmp/zm_birdnet_test.wav"))
    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_successful_extraction(self, mock_isfile, mock_run, mock_mkstemp, mock_close):
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-06-15 10:00:00",
            "Latitude": 43.65,
            "Longitude": -79.38,
        })
        mock_zm.event.return_value.path.return_value = "/events/123"

        # ffprobe finds audio
        probe_result = MagicMock(stdout="audio\n", stderr="")
        # ffmpeg succeeds
        ffmpeg_result = MagicMock()
        mock_run.side_effect = [probe_result, ffmpeg_result]

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)

        assert wav == "/tmp/zm_birdnet_test.wav"
        assert lat == 43.65
        assert lon == -79.38
        # June 15 = day 166, week = (166 // 7) + 1 = 24
        assert week == 24

    @patch("os.close")
    @patch("tempfile.mkstemp", return_value=(5, "/tmp/zm_birdnet_test.wav"))
    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_week_computed_correctly_for_january(self, mock_isfile, mock_run, mock_mkstemp, mock_close):
        """January 1 should give week 1."""
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-01-01 12:00:00",
            "Latitude": None,
            "Longitude": None,
        })
        mock_zm.event.return_value.path.return_value = "/events/1"

        probe_result = MagicMock(stdout="audio\n")
        ffmpeg_result = MagicMock()
        mock_run.side_effect = [probe_result, ffmpeg_result]

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 1)
        assert week == 1

    @patch("os.close")
    @patch("tempfile.mkstemp", return_value=(5, "/tmp/zm_birdnet_test.wav"))
    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_week_clamped_to_48(self, mock_isfile, mock_run, mock_mkstemp, mock_close):
        """Late December should clamp to week 48."""
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-12-31 12:00:00",
            "Latitude": None,
            "Longitude": None,
        })
        mock_zm.event.return_value.path.return_value = "/events/1"

        probe_result = MagicMock(stdout="audio\n")
        ffmpeg_result = MagicMock()
        mock_run.side_effect = [probe_result, ffmpeg_result]

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 1)
        assert week <= 48

    @patch("os.unlink")
    @patch("os.close")
    @patch("tempfile.mkstemp", return_value=(5, "/tmp/zm_birdnet_test.wav"))
    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_cleans_up_wav_on_ffmpeg_failure(self, mock_isfile, mock_run, mock_mkstemp, mock_close, mock_unlink):
        """Temp WAV file is cleaned up when ffmpeg fails."""
        from pyzm.ml.detector import Detector
        import subprocess

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-06-15 10:00:00",
            "Latitude": None,
            "Longitude": None,
        })
        mock_zm.event.return_value.path.return_value = "/events/123"

        # ffprobe finds audio
        probe_result = MagicMock(stdout="audio\n")
        # ffmpeg fails
        mock_run.side_effect = [
            probe_result,
            subprocess.CalledProcessError(1, "ffmpeg"),
        ]

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert wav is None
        mock_unlink.assert_called_once_with("/tmp/zm_birdnet_test.wav")

    @patch("os.close")
    @patch("tempfile.mkstemp", return_value=(5, "/tmp/zm_birdnet_test.wav"))
    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_monitor_latlon_defaults_to_negative(self, mock_isfile, mock_run, mock_mkstemp, mock_close):
        """Missing Latitude/Longitude default to -1.0."""
        from pyzm.ml.detector import Detector

        mock_zm = MagicMock()
        mock_zm._get_db.return_value = self._make_mock_db(row={
            "DefaultVideo": "video.mp4",
            "StartDateTime": "2025-06-15 10:00:00",
            "Latitude": None,
            "Longitude": None,
        })
        mock_zm.event.return_value.path.return_value = "/events/123"

        probe_result = MagicMock(stdout="audio\n")
        ffmpeg_result = MagicMock()
        mock_run.side_effect = [probe_result, ffmpeg_result]

        wav, week, lat, lon = Detector._extract_event_audio(mock_zm, 123)
        assert lat == -1.0
        assert lon == -1.0


# ===================================================================
# TestDetectEventAudioIntegration
# ===================================================================

class TestDetectEventAudioIntegration:
    """Tests for audio extraction wiring in detect_event()."""

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_event_sets_audio_context_when_audio_model_present(self, mock_pipeline_cls):
        """detect_event calls set_audio_context when audio models are in config."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult(detections=[_det("person")])
        mock_pipeline_cls.return_value = pipeline

        # Config with audio model
        config = DetectorConfig(
            models=[
                ModelConfig(name="YOLO", type=ModelType.OBJECT),
                ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET),
            ],
        )
        det = Detector(config=config)

        # Mock zm_client
        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([(1, MagicMock())], {"original": (480, 640)})

        with patch.object(Detector, "_extract_event_audio", return_value=("/tmp/audio.wav", 24, 43.0, -79.0)) as mock_extract:
            result = det.detect_event(mock_zm, 123)

        mock_extract.assert_called_once_with(mock_zm, 123)
        pipeline.set_audio_context.assert_called_once_with("/tmp/audio.wav", 24, 43.0, -79.0)

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_event_skips_audio_when_no_audio_model(self, mock_pipeline_cls):
        """detect_event does not extract audio when no audio model in config."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult(detections=[_det("person")])
        mock_pipeline_cls.return_value = pipeline

        # Config without audio model
        config = DetectorConfig(
            models=[ModelConfig(name="YOLO", type=ModelType.OBJECT)],
        )
        det = Detector(config=config)

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([(1, MagicMock())], {"original": (480, 640)})

        with patch.object(Detector, "_extract_event_audio") as mock_extract:
            det.detect_event(mock_zm, 123)

        mock_extract.assert_not_called()
        pipeline.set_audio_context.assert_not_called()

    @patch("os.unlink")
    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_event_cleans_up_wav_after_detection(self, mock_pipeline_cls, mock_unlink):
        """Temp WAV file is deleted after detection completes."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult(detections=[_det("person")])
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[
                ModelConfig(name="YOLO", type=ModelType.OBJECT),
                ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET),
            ],
        )
        det = Detector(config=config)

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([(1, MagicMock())], {"original": (480, 640)})

        with patch.object(Detector, "_extract_event_audio", return_value=("/tmp/audio.wav", 24, 43.0, -79.0)):
            det.detect_event(mock_zm, 123)

        mock_unlink.assert_called_once_with("/tmp/audio.wav")

    @patch("os.unlink")
    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_event_cleans_up_wav_even_on_error(self, mock_pipeline_cls, mock_unlink):
        """Temp WAV file is deleted even when detection raises."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.side_effect = RuntimeError("boom")
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[
                ModelConfig(name="YOLO", type=ModelType.OBJECT),
                ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET),
            ],
        )
        det = Detector(config=config)

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([(1, MagicMock())], {"original": (480, 640)})

        with patch.object(Detector, "_extract_event_audio", return_value=("/tmp/audio.wav", 24, 43.0, -79.0)):
            # The error in pipeline.run is caught per-frame in _detect_multi_frame,
            # so detect_event should not raise.
            det.detect_event(mock_zm, 123)

        mock_unlink.assert_called_once_with("/tmp/audio.wav")

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_event_no_cleanup_when_no_wav(self, mock_pipeline_cls):
        """No cleanup attempted when audio extraction returns None."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult()
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[
                ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET),
            ],
        )
        det = Detector(config=config)

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([(1, MagicMock())], {"original": (480, 640)})

        with patch.object(Detector, "_extract_event_audio", return_value=(None, -1, -1.0, -1.0)):
            with patch("os.unlink") as mock_unlink:
                det.detect_event(mock_zm, 123)

        mock_unlink.assert_not_called()

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_event_disabled_audio_model_not_triggering_extraction(self, mock_pipeline_cls):
        """Disabled audio models do not trigger audio extraction."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult()
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[
                ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET, enabled=False),
            ],
        )
        det = Detector(config=config)

        mock_zm = MagicMock()
        mock_zm.event.return_value.extract_frames.return_value = ([(1, MagicMock())], {"original": (480, 640)})

        with patch.object(Detector, "_extract_event_audio") as mock_extract:
            det.detect_event(mock_zm, 123)

        mock_extract.assert_not_called()


class TestDetectAudio:
    """Tests for Detector.detect_audio() convenience method."""

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_audio_sets_context_and_runs(self, mock_pipeline_cls):
        """detect_audio sets audio context on pipeline and runs with dummy image."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult(
            detections=[Detection(
                label="Robin", confidence=0.85,
                bbox=BBox(0, 0, 1, 1),
                model_name="BirdNET", detection_type="audio",
            )]
        )
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET)],
        )
        det = Detector(config=config)

        result = det.detect_audio("/tmp/birds.wav", event_week=20, lat=43.0, lon=-79.0)

        pipeline.set_audio_context.assert_called_once_with("/tmp/birds.wav", 20, 43.0, -79.0)
        pipeline.run.assert_called_once()
        assert result.frame_id == "audio"
        assert result.labels == ["Robin"]

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_audio_defaults(self, mock_pipeline_cls):
        """detect_audio passes defaults when optional args omitted."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult()
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET)],
        )
        det = Detector(config=config)

        det.detect_audio("/tmp/birds.wav")

        pipeline.set_audio_context.assert_called_once_with("/tmp/birds.wav", -1, -1.0, -1.0)

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_audio_empty_when_no_species(self, mock_pipeline_cls):
        """detect_audio returns empty result when no species detected."""
        from pyzm.ml.detector import Detector

        pipeline = MagicMock()
        pipeline._backends = []
        pipeline.run.return_value = DetectionResult()
        mock_pipeline_cls.return_value = pipeline

        config = DetectorConfig(
            models=[ModelConfig(name="BirdNET", type=ModelType.AUDIO, framework=ModelFramework.BIRDNET)],
        )
        det = Detector(config=config)

        result = det.detect_audio("/tmp/silence.wav")

        assert result.matched is False
        assert result.labels == []
        assert result.frame_id == "audio"
