"""E2E tests for ZoneMinder zone retrieval."""

from __future__ import annotations

import pytest

from pyzm.models.zm import Zone

pytestmark = pytest.mark.zm_e2e


class TestZones:
    def test_monitor_zones_returns_list(self, any_monitor):
        zones = any_monitor.get_zones()
        assert isinstance(zones, list)

    def test_zones_not_empty(self, any_monitor):
        zones = any_monitor.get_zones()
        if not zones:
            pytest.skip("Monitor has no zones configured")
        assert len(zones) > 0

    def test_zone_is_correct_type(self, any_monitor):
        zones = any_monitor.get_zones()
        if not zones:
            pytest.skip("Monitor has no zones configured")
        assert isinstance(zones[0], Zone)

    def test_zone_has_name(self, any_monitor):
        zones = any_monitor.get_zones()
        if not zones:
            pytest.skip("Monitor has no zones configured")
        assert isinstance(zones[0].name, str)
        assert len(zones[0].name) > 0

    def test_zone_points_are_valid(self, any_monitor):
        zones = any_monitor.get_zones()
        if not zones:
            pytest.skip("Monitor has no zones configured")
        points = zones[0].points
        assert isinstance(points, list)
        assert len(points) >= 3, "A zone polygon needs at least 3 points"
        for p in points:
            assert isinstance(p, tuple)
            assert len(p) == 2
            assert isinstance(p[0], int) and isinstance(p[1], int)
