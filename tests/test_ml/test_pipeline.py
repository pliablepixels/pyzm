"""Tests for pyzm.ml.pipeline -- ModelPipeline orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import (
    DetectorConfig,
    MatchStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
    TypeOverrides,
)
from pyzm.models.detection import BBox, Detection, DetectionResult


# ===================================================================
# Helpers
# ===================================================================

def _det(label: str, x1: int = 10, y1: int = 10, x2: int = 50, y2: int = 50, conf: float = 0.9, model: str = "test") -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), model_name=model)


def _make_model_config(name: str = "test", mtype: ModelType = ModelType.OBJECT, enabled: bool = True, pre_existing: list[str] | None = None) -> ModelConfig:
    return ModelConfig(
        name=name,
        enabled=enabled,
        type=mtype,
        framework=ModelFramework.OPENCV,
        min_confidence=0.3,
        pattern=".*",
        pre_existing_labels=pre_existing or [],
    )


def _make_mock_backend(name: str, detections: list[Detection]):
    backend = MagicMock()
    backend.name = name
    backend.detect.return_value = detections
    backend.load.return_value = None
    backend.is_loaded = True
    return backend


# ===================================================================
# TestModelPipeline
# ===================================================================

class TestModelPipeline:
    """Tests for ModelPipeline with mocked backends."""

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_single_model_returns_detections(self, mock_zone_filter, mock_create):
        """Single model returns detections."""
        dets = [_det("person"), _det("car")]

        mc = _make_model_config("yolov4")
        mock_backend = _make_mock_backend("yolov4", dets)
        mock_create.return_value = mock_backend

        # Zone filter passes everything through
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc], match_strategy=MatchStrategy.FIRST)

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        # Create a mock image
        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)
        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 2
        assert result.detections[0].label == "person"

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_first_strategy_stops_at_first_match(self, mock_zone_filter, mock_create):
        """FIRST strategy: stops after first backend returns detections."""
        mc1 = _make_model_config("model1")
        mc2 = _make_model_config("model2")

        backend1 = _make_mock_backend("model1", [_det("person", model="model1")])
        backend2 = _make_mock_backend("model2", [_det("car", model="model2"), _det("truck", model="model2")])

        mock_create.side_effect = [backend1, backend2]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.FIRST,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # Should return model1's results (FIRST match)
        assert len(result.detections) == 1
        assert result.detections[0].label == "person"
        # model2 should NOT have been called
        backend2.detect.assert_not_called()

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_most_strategy_picks_most_detections(self, mock_zone_filter, mock_create):
        """MOST strategy: picks the variant with the most detections."""
        mc1 = _make_model_config("model1")
        mc2 = _make_model_config("model2")

        backend1 = _make_mock_backend("model1", [_det("person", model="model1")])
        backend2 = _make_mock_backend("model2", [_det("car", model="model2"), _det("truck", model="model2")])

        mock_create.side_effect = [backend1, backend2]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.MOST,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # Should return model2's results (2 > 1)
        assert len(result.detections) == 2

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_union_strategy_combines_results(self, mock_zone_filter, mock_create):
        """UNION strategy: combines all results."""
        mc1 = _make_model_config("model1")
        mc2 = _make_model_config("model2")

        backend1 = _make_mock_backend("model1", [_det("person", model="model1")])
        backend2 = _make_mock_backend("model2", [_det("car", model="model2")])

        mock_create.side_effect = [backend1, backend2]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.UNION,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # Should combine: 1 from model1 + 1 from model2
        assert len(result.detections) == 2
        labels = {d.label for d in result.detections}
        assert labels == {"person", "car"}

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_pre_existing_labels_gate(self, mock_zone_filter, mock_create):
        """Face model should be skipped if 'person' not detected by object model."""
        mc_obj = _make_model_config("yolov4", mtype=ModelType.OBJECT)
        mc_face = _make_model_config("dlib", mtype=ModelType.FACE, pre_existing=["person"])

        # Object model returns car only (no person)
        backend_obj = _make_mock_backend("yolov4", [_det("car", model="yolov4")])
        backend_face = _make_mock_backend("dlib", [_det("John", model="dlib")])

        mock_create.side_effect = [backend_obj, backend_face]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc_obj, mc_face],
            match_strategy=MatchStrategy.FIRST,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # Face model should not have been called since 'person' was not detected
        backend_face.detect.assert_not_called()
        assert len(result.detections) == 1
        assert result.detections[0].label == "car"

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_pre_existing_labels_satisfied(self, mock_zone_filter, mock_create):
        """Face model should run when 'person' IS detected by object model."""
        mc_obj = _make_model_config("yolov4", mtype=ModelType.OBJECT)
        mc_face = _make_model_config("dlib", mtype=ModelType.FACE, pre_existing=["person"])

        # Object model returns person
        backend_obj = _make_mock_backend("yolov4", [_det("person", model="yolov4")])
        backend_face = _make_mock_backend("dlib", [_det("John", model="dlib")])

        mock_create.side_effect = [backend_obj, backend_face]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc_obj, mc_face],
            match_strategy=MatchStrategy.FIRST,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # Both backends should have been called
        backend_face.detect.assert_called_once()
        labels = {d.label for d in result.detections}
        assert "person" in labels or "John" in labels

    @patch("pyzm.ml.pipeline._create_backend")
    def test_disabled_model_is_skipped(self, mock_create):
        """Disabled models should not be loaded or run."""
        mc_enabled = _make_model_config("yolov4", enabled=True)
        mc_disabled = _make_model_config("slow_model", enabled=False)

        backend = _make_mock_backend("yolov4", [_det("person")])
        mock_create.return_value = backend

        config = DetectorConfig(
            models=[mc_enabled, mc_disabled],
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.load()

        # Only one backend should have been created
        assert mock_create.call_count == 1

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_backend_exception_handled_gracefully(self, mock_zone_filter, mock_create):
        """Backend exceptions should be caught and logged, not propagated."""
        mc1 = _make_model_config("bad_model")
        mc2 = _make_model_config("good_model")

        bad_backend = _make_mock_backend("bad_model", [])
        bad_backend.detect.side_effect = RuntimeError("GPU out of memory")

        good_backend = _make_mock_backend("good_model", [_det("person")])

        mock_create.side_effect = [bad_backend, good_backend]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.UNION,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        # Should NOT raise
        result = pipeline.run(mock_image)
        # Should still get results from the good model
        assert len(result.detections) == 1

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_most_unique_strategy(self, mock_zone_filter, mock_create):
        """MOST_UNIQUE strategy: picks variant with most unique labels."""
        mc1 = _make_model_config("model1")
        mc2 = _make_model_config("model2")

        # model1: 3 persons (1 unique label)
        backend1 = _make_mock_backend("model1", [_det("person"), _det("person"), _det("person")])
        # model2: person + car (2 unique labels)
        backend2 = _make_mock_backend("model2", [_det("person"), _det("car")])

        mock_create.side_effect = [backend1, backend2]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.MOST_UNIQUE,
        )

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)
        # model2 has more unique labels (2 > 1)
        assert len(result.detections) == 2


class TestZoneRescaling:
    """Tests for zone polygon rescaling when images are resized."""

    @patch("pyzm.ml.pipeline._create_backend")
    def test_zones_rescaled_when_image_smaller_than_original(self, mock_create):
        """Zone polygons should be rescaled from original to resized coords."""
        from pyzm.ml.pipeline import ModelPipeline
        from pyzm.models.zm import Zone

        mc = _make_model_config("test")
        backend = _make_mock_backend("test", [_det("person", x1=10, y1=10, x2=100, y2=100)])
        mock_create.return_value = backend

        config = DetectorConfig(models=[mc])
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (240, 400, 3)  # resized: 400x240

        # Zone defined in original coords (800x480)
        zone = Zone(name="front_door", points=[(0, 0), (800, 0), (800, 480), (0, 480)])

        result = pipeline.run(mock_image, zones=[zone], original_shape=(480, 800))

        # The image_dimensions should reflect original and resized
        assert result.image_dimensions["original"] == (480, 800)
        assert result.image_dimensions["resized"] == (240, 400)

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_zone_points_rescaled_correctly(self, mock_zone_filter, mock_create):
        """Verify the actual zone point coordinates after rescaling."""
        from pyzm.ml.pipeline import ModelPipeline
        from pyzm.models.zm import Zone

        mc = _make_model_config("test")
        backend = _make_mock_backend("test", [_det("person")])
        mock_create.return_value = backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc])
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (240, 400, 3)  # resized from 480x800

        zone = Zone(
            name="test_zone",
            points=[(100, 200), (400, 200), (400, 400), (100, 400)],
            pattern="person",
        )

        pipeline.run(mock_image, zones=[zone], original_shape=(480, 800))

        # Verify filter_by_zone was called with rescaled zone points
        call_args = mock_zone_filter.call_args
        zone_dicts = call_args[0][1]
        assert len(zone_dicts) == 1
        pts = zone_dicts[0]["points"]
        # xfactor = 400/800 = 0.5, yfactor = 240/480 = 0.5
        assert pts == [(50, 100), (200, 100), (200, 200), (50, 200)]
        assert zone_dicts[0]["name"] == "test_zone"
        assert zone_dicts[0]["pattern"] == "person"

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_no_rescaling_when_no_original_shape(self, mock_zone_filter, mock_create):
        """Without original_shape, zone points should pass through unchanged."""
        from pyzm.ml.pipeline import ModelPipeline
        from pyzm.models.zm import Zone

        mc = _make_model_config("test")
        backend = _make_mock_backend("test", [_det("person")])
        mock_create.return_value = backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc])
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (480, 800, 3)

        zone = Zone(name="z", points=[(100, 200), (400, 200), (400, 400), (100, 400)])

        pipeline.run(mock_image, zones=[zone])

        zone_dicts = mock_zone_filter.call_args[0][1]
        assert zone_dicts[0]["points"] == [(100, 200), (400, 200), (400, 400), (100, 400)]

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_no_rescaling_when_shapes_match(self, mock_zone_filter, mock_create):
        """If original_shape equals image shape, no rescaling should occur."""
        from pyzm.ml.pipeline import ModelPipeline
        from pyzm.models.zm import Zone

        mc = _make_model_config("test")
        backend = _make_mock_backend("test", [_det("person")])
        mock_create.return_value = backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc])
        pipeline = ModelPipeline(config)

        mock_image = MagicMock()
        mock_image.shape = (480, 800, 3)

        zone = Zone(name="z", points=[(100, 200), (400, 200)])

        pipeline.run(mock_image, zones=[zone], original_shape=(480, 800))

        zone_dicts = mock_zone_filter.call_args[0][1]
        assert zone_dicts[0]["points"] == [(100, 200), (400, 200)]
        # image_dimensions should reflect no resize
        assert mock_zone_filter.call_args is not None


class TestPerTypeConfig:
    """Tests for per-type config overrides in the pipeline."""

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_per_type_match_strategy(self, mock_zone_filter, mock_create):
        """Object type uses MOST, face type uses UNION via type_overrides."""
        from pyzm.ml.pipeline import ModelPipeline

        mc_obj1 = _make_model_config("obj1", mtype=ModelType.OBJECT)
        mc_obj2 = _make_model_config("obj2", mtype=ModelType.OBJECT)
        mc_face1 = _make_model_config("face1", mtype=ModelType.FACE)
        mc_face2 = _make_model_config("face2", mtype=ModelType.FACE)

        # obj1: 1 detection, obj2: 2 detections → MOST picks obj2
        backend_obj1 = _make_mock_backend("obj1", [_det("person", model="obj1")])
        backend_obj2 = _make_mock_backend("obj2", [_det("car", model="obj2"), _det("truck", model="obj2")])
        # face1: 1 detection, face2: 1 detection → UNION combines both
        backend_face1 = _make_mock_backend("face1", [_det("Alice", model="face1")])
        backend_face2 = _make_mock_backend("face2", [_det("Bob", model="face2")])

        mock_create.side_effect = [backend_obj1, backend_obj2, backend_face1, backend_face2]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc_obj1, mc_obj2, mc_face1, mc_face2],
            match_strategy=MatchStrategy.FIRST,  # global default
            type_overrides={
                ModelType.OBJECT: TypeOverrides(match_strategy=MatchStrategy.MOST),
                ModelType.FACE: TypeOverrides(match_strategy=MatchStrategy.UNION),
            },
        )

        pipeline = ModelPipeline(config)
        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)
        labels = [d.label for d in result.detections]

        # MOST for object: obj2 wins (2 > 1) → car, truck
        assert "car" in labels
        assert "truck" in labels
        assert "person" not in labels
        # UNION for face: both → Alice, Bob
        assert "Alice" in labels
        assert "Bob" in labels

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    @patch("pyzm.ml.pipeline.load_past_detections")
    @patch("pyzm.ml.pipeline.save_past_detections")
    def test_per_type_match_past_detections(
        self, mock_save, mock_load, mock_zone_filter, mock_create,
    ):
        """match_past_detections enabled for object but disabled for face."""
        from pyzm.ml.pipeline import ModelPipeline

        mc_obj = _make_model_config("yolo", mtype=ModelType.OBJECT)
        mc_face = _make_model_config("dlib", mtype=ModelType.FACE)

        # Object backend returns "person" at same location as saved
        backend_obj = _make_mock_backend("yolo", [
            Detection(label="person", confidence=0.9,
                      bbox=BBox(x1=10, y1=10, x2=50, y2=50),
                      model_name="yolo", detection_type="object"),
        ])
        # Face backend returns "Alice"
        backend_face = _make_mock_backend("dlib", [
            Detection(label="Alice", confidence=0.8,
                      bbox=BBox(x1=10, y1=10, x2=50, y2=50),
                      model_name="dlib", detection_type="face"),
        ])

        mock_create.side_effect = [backend_obj, backend_face]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        # Saved past detection: "person" at exact same bbox
        mock_load.return_value = ([[10, 10, 50, 50]], ["person"])

        config = DetectorConfig(
            models=[mc_obj, mc_face],
            match_strategy=MatchStrategy.FIRST,
            match_past_detections=False,  # global default: off
            image_path="/tmp",
            type_overrides={
                ModelType.OBJECT: TypeOverrides(match_past_detections=True),
                # face: no override → uses global (False)
            },
        )

        pipeline = ModelPipeline(config)
        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)
        labels = [d.label for d in result.detections]

        # "person" should be filtered out (object type has past-detection enabled, matches saved)
        assert "person" not in labels
        # "Alice" should be kept (face type has past-detection disabled)
        assert "Alice" in labels


# ===================================================================
# TestAudioContext
# ===================================================================

class TestAudioContext:
    """Tests for audio context and audio model dispatch in the pipeline."""

    def test_set_audio_context(self):
        """set_audio_context stores audio metadata on the pipeline."""
        from pyzm.ml.pipeline import ModelPipeline

        config = DetectorConfig()
        pipeline = ModelPipeline(config)

        assert pipeline._audio_path is None
        assert pipeline._audio_week == -1

        pipeline.set_audio_context("/tmp/audio.wav", event_week=20, monitor_lat=43.0, monitor_lon=-79.0)

        assert pipeline._audio_path == "/tmp/audio.wav"
        assert pipeline._audio_week == 20
        assert pipeline._monitor_lat == 43.0
        assert pipeline._monitor_lon == -79.0

    def test_audio_context_defaults(self):
        """Pipeline starts with no audio context."""
        from pyzm.ml.pipeline import ModelPipeline

        config = DetectorConfig()
        pipeline = ModelPipeline(config)

        assert pipeline._audio_path is None
        assert pipeline._audio_week == -1
        assert pipeline._monitor_lat == -1.0
        assert pipeline._monitor_lon == -1.0

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_audio_backend_calls_detect_audio(self, mock_zone_filter, mock_create):
        """Audio model type dispatches to detect_audio instead of detect."""
        from pyzm.ml.pipeline import ModelPipeline

        audio_dets = [Detection(
            label="American Robin", confidence=0.9,
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            model_name="BirdNET", detection_type="audio",
        )]

        mc = ModelConfig(
            name="BirdNET", type=ModelType.AUDIO,
            framework=ModelFramework.BIRDNET,
        )
        mock_backend = _make_mock_backend("BirdNET", [])
        mock_backend.detect_audio = MagicMock(return_value=audio_dets)
        mock_create.return_value = mock_backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc], match_strategy=MatchStrategy.FIRST)
        pipeline = ModelPipeline(config)
        pipeline.set_audio_context("/tmp/audio.wav", event_week=20, monitor_lat=43.0, monitor_lon=-79.0)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # detect(image) should NOT have been called
        mock_backend.detect.assert_not_called()
        # detect_audio should have been called with audio context
        mock_backend.detect_audio.assert_called_once_with(
            "/tmp/audio.wav", 20, 43.0, -79.0,
        )
        assert len(result.detections) == 1
        assert result.detections[0].label == "American Robin"
        assert result.detections[0].detection_type == "audio"

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_audio_backend_skipped_when_no_audio_path(self, mock_zone_filter, mock_create):
        """Audio model is skipped when no audio context is set."""
        from pyzm.ml.pipeline import ModelPipeline

        mc = ModelConfig(
            name="BirdNET", type=ModelType.AUDIO,
            framework=ModelFramework.BIRDNET,
        )
        mock_backend = _make_mock_backend("BirdNET", [])
        mock_backend.detect_audio = MagicMock()
        mock_create.return_value = mock_backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc], match_strategy=MatchStrategy.FIRST)
        pipeline = ModelPipeline(config)
        # No set_audio_context call — audio_path stays None

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        mock_backend.detect.assert_not_called()
        mock_backend.detect_audio.assert_not_called()
        assert result.detections == []

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_mixed_object_and_audio_models(self, mock_zone_filter, mock_create):
        """Object and audio models run together, both contribute detections."""
        from pyzm.ml.pipeline import ModelPipeline

        obj_dets = [_det("person", model="yolo")]
        audio_dets = [Detection(
            label="American Robin", confidence=0.85,
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            model_name="BirdNET", detection_type="audio",
        )]

        mc_obj = _make_model_config("yolo", mtype=ModelType.OBJECT)
        mc_audio = ModelConfig(
            name="BirdNET", type=ModelType.AUDIO,
            framework=ModelFramework.BIRDNET,
        )

        backend_obj = _make_mock_backend("yolo", obj_dets)
        backend_audio = _make_mock_backend("BirdNET", [])
        backend_audio.detect_audio = MagicMock(return_value=audio_dets)

        mock_create.side_effect = [backend_obj, backend_audio]
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(
            models=[mc_obj, mc_audio],
            match_strategy=MatchStrategy.FIRST,
        )
        pipeline = ModelPipeline(config)
        pipeline.set_audio_context("/tmp/audio.wav", event_week=20)

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        labels = {d.label for d in result.detections}
        assert "person" in labels
        assert "American Robin" in labels
        backend_obj.detect.assert_called_once()
        backend_audio.detect_audio.assert_called_once()

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_audio_backend_exception_handled_gracefully(self, mock_zone_filter, mock_create):
        """Exception in detect_audio is caught and logged, not propagated."""
        from pyzm.ml.pipeline import ModelPipeline

        mc = ModelConfig(
            name="BirdNET", type=ModelType.AUDIO,
            framework=ModelFramework.BIRDNET,
        )
        mock_backend = _make_mock_backend("BirdNET", [])
        mock_backend.detect_audio = MagicMock(side_effect=RuntimeError("BirdNET crashed"))
        mock_create.return_value = mock_backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[mc], match_strategy=MatchStrategy.FIRST)
        pipeline = ModelPipeline(config)
        pipeline.set_audio_context("/tmp/audio.wav")

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        # Should NOT raise
        result = pipeline.run(mock_image)
        assert result.detections == []
