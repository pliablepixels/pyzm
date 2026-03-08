"""Tests for pyzm.ml.filters."""

from pyzm.ml.filters import filter_by_zone
from pyzm.models.detection import BBox, Detection


class TestIgnorePatternMultiZone:
    """H10: ignore_pattern in zone A should not prevent matching in zone B."""

    def _det(self, label="car"):
        return Detection(label=label, confidence=0.9,
                        bbox=BBox(x1=10, y1=10, x2=90, y2=90),
                        model_name="test", detection_type="object")

    def test_kept_if_second_zone_allows(self):
        zones = [
            {"name": "a", "points": [(0,0),(100,0),(100,100),(0,100)],
             "pattern": ".*", "ignore_pattern": "car"},
            {"name": "b", "points": [(0,0),(100,0),(100,100),(0,100)],
             "pattern": ".*", "ignore_pattern": None},
        ]
        kept, errors = filter_by_zone([self._det()], zones, (100, 100))
        assert len(kept) == 1

    def test_dropped_if_all_zones_ignore(self):
        zones = [
            {"name": "a", "points": [(0,0),(100,0),(100,100),(0,100)],
             "pattern": ".*", "ignore_pattern": "car"},
            {"name": "b", "points": [(0,0),(100,0),(100,100),(0,100)],
             "pattern": ".*", "ignore_pattern": "car"},
        ]
        kept, errors = filter_by_zone([self._det()], zones, (100, 100))
        assert len(kept) == 0

    def test_no_zones_passes_everything(self):
        kept, errors = filter_by_zone([self._det()], [], (100, 100))
        assert len(kept) == 1


class TestFilterPastPerType:
    def test_filters_duplicate_detections(self, tmp_path):
        import pickle
        from pyzm.models.config import DetectorConfig
        from pyzm.ml.filters import filter_past_per_type

        # Seed a past detection file
        past_file = str(tmp_path / "past_detections.pkl")
        with open(past_file, "wb") as f:
            pickle.dump([[10, 10, 50, 50]], f)
            pickle.dump(["person"], f)

        dets = [Detection(label="person", confidence=0.9, bbox=BBox(10, 10, 50, 50), model_name="yolo", detection_type="object")]
        cfg = DetectorConfig(match_past_detections=True, past_det_max_diff_area="10%", image_path=str(tmp_path))

        result = filter_past_per_type(dets, cfg)
        assert len(result) == 0  # duplicate filtered out

    def test_passes_new_detections(self, tmp_path):
        from pyzm.models.config import DetectorConfig
        from pyzm.ml.filters import filter_past_per_type

        dets = [Detection(label="car", confidence=0.8, bbox=BBox(100, 100, 200, 200), model_name="yolo", detection_type="object")]
        cfg = DetectorConfig(match_past_detections=True, image_path=str(tmp_path))

        result = filter_past_per_type(dets, cfg)
        assert len(result) == 1

    def test_disabled_returns_all(self, tmp_path):
        from pyzm.models.config import DetectorConfig
        from pyzm.ml.filters import filter_past_per_type

        dets = [Detection(label="person", confidence=0.9, bbox=BBox(10, 10, 50, 50), model_name="yolo")]
        cfg = DetectorConfig(match_past_detections=False, image_path=str(tmp_path))

        result = filter_past_per_type(dets, cfg)
        assert len(result) == 1
