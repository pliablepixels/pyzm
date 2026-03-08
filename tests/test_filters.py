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
