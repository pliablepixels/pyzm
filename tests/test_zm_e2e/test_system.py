"""E2E tests for system health endpoints (readonly)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.zm_e2e


class TestSystem:
    def test_is_running(self, zm_client):
        """is_running() should return True on a live server."""
        assert zm_client.is_running() is True

    def test_system_load(self, zm_client):
        """system_load() should return 1/5/15 min averages."""
        load = zm_client.system_load()
        assert "1min" in load
        assert "5min" in load
        assert "15min" in load
        for val in load.values():
            assert isinstance(val, float)
            assert val >= 0

    def test_disk_usage(self, zm_client):
        """disk_usage() should return a non-empty dict."""
        usage = zm_client.disk_usage()
        assert isinstance(usage, dict)

    def test_timezone(self, zm_client):
        """timezone() should return a non-empty string."""
        tz = zm_client.timezone()
        assert isinstance(tz, str)
        assert len(tz) > 0
