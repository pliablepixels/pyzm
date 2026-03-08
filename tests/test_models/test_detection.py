"""Tests for pyzm.models.detection -- detection result models."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pyzm.models.detection import BBox, Detection, DetectionResult


# ===================================================================
# TestBBox
# ===================================================================

class TestBBox:
    """Tests for the BBox frozen dataclass."""

    def test_creation(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.x1 == 10
        assert bb.y1 == 20
        assert bb.x2 == 50
        assert bb.y2 == 80

    def test_width(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.width == 40

    def test_height(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.height == 60

    def test_area(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.area == 40 * 60  # 2400

    def test_center(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.center == (30, 50)

    def test_center_odd_values(self):
        bb = BBox(x1=0, y1=0, x2=99, y2=99)
        assert bb.center == (49, 49)

    def test_as_list(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.as_list() == [10, 20, 50, 80]

    def test_as_polygon_coords(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        coords = bb.as_polygon_coords()
        assert coords == [
            (10, 20),  # top-left
            (50, 20),  # top-right
            (50, 80),  # bottom-right
            (10, 80),  # bottom-left
        ]

    def test_frozen_immutability(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        with pytest.raises(AttributeError):
            bb.x1 = 100

    def test_frozen_immutability_y2(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        with pytest.raises(AttributeError):
            bb.y2 = 200

    def test_zero_area_bbox(self):
        bb = BBox(x1=10, y1=10, x2=10, y2=10)
        assert bb.area == 0
        assert bb.width == 0
        assert bb.height == 0


# ===================================================================
# TestDetection
# ===================================================================

class TestDetection:
    """Tests for the Detection frozen dataclass."""

    def test_creation(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        det = Detection(label="person", confidence=0.95, bbox=bb)
        assert det.label == "person"
        assert det.confidence == 0.95
        assert det.bbox == bb
        assert det.model_name == ""
        assert det.detection_type == "object"

    def test_creation_with_all_fields(self):
        bb = BBox(x1=0, y1=0, x2=100, y2=100)
        det = Detection(
            label="face_john",
            confidence=0.88,
            bbox=bb,
            model_name="dlib",
            detection_type="face",
        )
        assert det.model_name == "dlib"
        assert det.detection_type == "face"

    def test_matches_pattern_simple(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        assert det.matches_pattern("person") is True
        assert det.matches_pattern("car") is False

    def test_matches_pattern_regex(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        assert det.matches_pattern("(person|car)") is True
        assert det.matches_pattern("per.*") is True
        assert det.matches_pattern("^per") is True
        assert det.matches_pattern("^car") is False

    def test_matches_pattern_wildcard(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        assert det.matches_pattern(".*") is True

    def test_frozen_immutability(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        with pytest.raises(AttributeError):
            det.label = "car"


# ===================================================================
# TestDetectionResult
# ===================================================================

class TestDetectionResult:
    """Tests for the DetectionResult aggregate dataclass."""

    def _make_detections(self):
        return [
            Detection(
                label="person", confidence=0.97,
                bbox=BBox(x1=10, y1=20, x2=50, y2=80),
                model_name="yolov4",
            ),
            Detection(
                label="car", confidence=0.85,
                bbox=BBox(x1=60, y1=30, x2=90, y2=70),
                model_name="yolov4",
            ),
        ]

    def test_matched_true(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.matched is True

    def test_matched_false_empty(self):
        dr = DetectionResult()
        assert dr.matched is False

    def test_labels_property(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.labels == ["person", "car"]

    def test_boxes_property(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.boxes == [[10, 20, 50, 80], [60, 30, 90, 70]]

    def test_confidences_property(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.confidences == [0.97, 0.85]

    def test_summary(self):
        dr = DetectionResult(detections=self._make_detections())
        summary = dr.summary
        assert "person:97%" in summary
        assert "car:85%" in summary

    def test_summary_empty(self):
        dr = DetectionResult()
        assert dr.summary == ""

    def test_filter_by_pattern_keeps_matching(self):
        dr = DetectionResult(detections=self._make_detections(), frame_id=1)
        filtered = dr.filter_by_pattern("person")
        assert len(filtered.detections) == 1
        assert filtered.detections[0].label == "person"
        assert filtered.frame_id == 1

    def test_filter_by_pattern_removes_nonmatching(self):
        dr = DetectionResult(detections=self._make_detections())
        filtered = dr.filter_by_pattern("truck")
        assert len(filtered.detections) == 0

    def test_filter_by_pattern_regex(self):
        dr = DetectionResult(detections=self._make_detections())
        filtered = dr.filter_by_pattern("(person|car)")
        assert len(filtered.detections) == 2

    def test_to_dict(self):
        dets = self._make_detections()
        dr = DetectionResult(
            detections=dets,
            frame_id=5,
            image_dimensions={"original": (480, 640)},
        )
        d = dr.to_dict()
        assert d["labels"] == ["person", "car"]
        assert d["boxes"] == [[10, 20, 50, 80], [60, 30, 90, 70]]
        assert d["confidences"] == [0.97, 0.85]
        assert d["frame_id"] == 5
        assert d["image_dimensions"] == {"original": (480, 640)}
        assert d["model_names"] == ["yolov4", "yolov4"]
        assert d["error_boxes"] == []
        assert d["polygons"] == []

    def test_to_dict_with_error_boxes(self):
        eb = BBox(x1=0, y1=0, x2=10, y2=10)
        dr = DetectionResult(
            detections=self._make_detections(),
            error_boxes=[eb],
        )
        d = dr.to_dict()
        assert d["error_boxes"] == [[0, 0, 10, 10]]

    def test_annotate_no_image_raises(self):
        dr = DetectionResult(detections=self._make_detections())
        with pytest.raises(ValueError, match="No image"):
            dr.annotate()

    # -- from_dict (round-trip) -------------------------------------------------

    def test_from_dict_round_trip(self):
        """to_dict() -> from_dict() reproduces detections."""
        dets = self._make_detections()
        eb = BBox(x1=0, y1=0, x2=5, y2=5)
        original = DetectionResult(
            detections=dets,
            frame_id=7,
            image_dimensions={"original": (480, 640)},
            error_boxes=[eb],
        )
        wire = original.to_dict()
        # Remove image from wire (from_dict always sets image=None)
        wire.pop("image", None)

        restored = DetectionResult.from_dict(wire)
        assert restored.labels == original.labels
        assert restored.confidences == original.confidences
        assert restored.boxes == original.boxes
        assert restored.frame_id == 7
        assert restored.image_dimensions == {"original": (480, 640)}
        assert len(restored.error_boxes) == 1
        assert restored.error_boxes[0].as_list() == [0, 0, 5, 5]
        assert restored.image is None

    def test_from_dict_empty(self):
        """from_dict({}) produces an empty DetectionResult."""
        dr = DetectionResult.from_dict({})
        assert dr.matched is False
        assert dr.labels == []
        assert dr.error_boxes == []
        assert dr.frame_id is None

    def test_from_dict_preserves_model_names(self):
        """model_names are restored correctly."""
        wire = {
            "labels": ["person", "car"],
            "boxes": [[10, 20, 50, 80], [60, 30, 90, 70]],
            "confidences": [0.97, 0.85],
            "model_names": ["yolov4", "yolov7"],
        }
        dr = DetectionResult.from_dict(wire)
        assert [d.model_name for d in dr.detections] == ["yolov4", "yolov7"]

    def test_from_dict_error_boxes(self):
        """error_boxes are reconstructed as BBox objects."""
        wire = {"error_boxes": [[0, 0, 10, 10], [5, 5, 20, 20]]}
        dr = DetectionResult.from_dict(wire)
        assert len(dr.error_boxes) == 2
        assert dr.error_boxes[0] == BBox(0, 0, 10, 10)
        assert dr.error_boxes[1] == BBox(5, 5, 20, 20)

    @pytest.mark.integration
    def test_annotate_with_mock_cv2(self):
        """Test annotate() by mocking cv2 to avoid requiring it."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.getTextSize.return_value = ((100, 20), 0)

        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_image.copy.return_value = mock_image
        mock_image.shape = (100, 100, 3)

        dr = DetectionResult(
            detections=self._make_detections(),
            image=mock_image,
        )

        with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": mock_np}):
            result = dr.annotate()
            assert mock_cv2.rectangle.called
            assert mock_cv2.putText.called

    @pytest.mark.integration
    def test_annotate_polygons(self):
        """annotate() draws zone polygons when provided."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.getTextSize.return_value = ((100, 20), 0)
        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_image.copy.return_value = mock_image

        dr = DetectionResult(detections=self._make_detections(), image=mock_image)
        polygons = [{"name": "yard", "value": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": mock_np}):
            dr.annotate(polygons=polygons, poly_color=(0, 255, 0), poly_thickness=2)
            assert mock_cv2.polylines.called

    @pytest.mark.integration
    def test_annotate_write_conf_false(self):
        """annotate(write_conf=False) omits confidence from labels."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.getTextSize.return_value = ((100, 20), 0)
        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_image.copy.return_value = mock_image

        dr = DetectionResult(detections=self._make_detections(), image=mock_image)

        with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": mock_np}):
            dr.annotate(write_conf=False)
            # Check that putText was called with label only (no %)
            for call in mock_cv2.putText.call_args_list:
                label_arg = call[0][1]
                assert "%" not in label_arg

    @pytest.mark.integration
    def test_annotate_draw_error_boxes_false(self):
        """annotate(draw_error_boxes=False) skips error box drawing."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.getTextSize.return_value = ((100, 20), 0)
        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_image.copy.return_value = mock_image

        eb = BBox(x1=0, y1=0, x2=10, y2=10)
        dr = DetectionResult(
            detections=self._make_detections(),
            image=mock_image,
            error_boxes=[eb],
        )

        with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": mock_np}):
            dr.annotate(draw_error_boxes=False)
            # rectangle called for detection boxes only (2 detections × 2 calls each),
            # NOT for error boxes
            # Each detection: 1 box rectangle + 1 label background rectangle = 2
            assert mock_cv2.rectangle.call_count == 4  # 2 detections × 2

    @pytest.mark.integration
    def test_annotate_draw_error_boxes_default(self):
        """By default, annotate() draws error boxes."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.getTextSize.return_value = ((100, 20), 0)
        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_image.copy.return_value = mock_image

        eb = BBox(x1=0, y1=0, x2=10, y2=10)
        dr = DetectionResult(
            detections=self._make_detections(),
            image=mock_image,
            error_boxes=[eb],
        )

        with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": mock_np}):
            dr.annotate()
            # 2 detections × 2 rectangles + 1 error box = 5
            assert mock_cv2.rectangle.call_count == 5


# ===================================================================
# TestDetectionTypeWireFormat
# ===================================================================

class TestDetectionTypeWireFormat:
    """Tests that detection_type survives the wire format round-trip."""

    def test_to_dict_includes_detection_types(self):
        """to_dict() emits a detection_types list matching detections order."""
        dets = [
            Detection(
                label="person", confidence=0.95,
                bbox=BBox(10, 20, 50, 80),
                model_name="yolov4", detection_type="object",
            ),
            Detection(
                label="john", confidence=0.88,
                bbox=BBox(60, 30, 90, 70),
                model_name="dlib", detection_type="face",
            ),
            Detection(
                label="ABC123", confidence=0.76,
                bbox=BBox(100, 40, 200, 60),
                model_name="openalpr", detection_type="alpr",
            ),
        ]
        dr = DetectionResult(detections=dets, frame_id=1)
        wire = dr.to_dict()
        assert "detection_types" in wire
        assert wire["detection_types"] == ["object", "face", "alpr"]

    def test_from_dict_restores_detection_types(self):
        """from_dict() restores detection_type on each Detection."""
        wire = {
            "labels": ["person", "john"],
            "boxes": [[10, 20, 50, 80], [60, 30, 90, 70]],
            "confidences": [0.95, 0.88],
            "model_names": ["yolov4", "dlib"],
            "detection_types": ["object", "face"],
        }
        dr = DetectionResult.from_dict(wire)
        assert dr.detections[0].detection_type == "object"
        assert dr.detections[1].detection_type == "face"

    def test_from_dict_defaults_detection_type_when_missing(self):
        """from_dict() defaults detection_type to 'object' when key is absent."""
        wire = {
            "labels": ["person", "car"],
            "boxes": [[10, 20, 50, 80], [60, 30, 90, 70]],
            "confidences": [0.95, 0.85],
            "model_names": ["yolov4", "yolov4"],
        }
        dr = DetectionResult.from_dict(wire)
        assert dr.detections[0].detection_type == "object"
        assert dr.detections[1].detection_type == "object"
