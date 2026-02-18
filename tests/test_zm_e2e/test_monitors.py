"""E2E tests for ZoneMinder monitor operations."""

from __future__ import annotations

import pytest

from pyzm.models.zm import Monitor

pytestmark = pytest.mark.zm_e2e


class TestMonitors:
    def test_monitors_returns_list(self, zm_client):
        monitors = zm_client.monitors()
        assert isinstance(monitors, list)
        assert len(monitors) > 0, "Expected at least one monitor"

    def test_monitor_is_correct_type(self, any_monitor):
        assert isinstance(any_monitor, Monitor)

    def test_monitor_id_is_int(self, any_monitor):
        assert isinstance(any_monitor.id, int)
        assert any_monitor.id > 0

    def test_monitor_name_is_str(self, any_monitor):
        assert isinstance(any_monitor.name, str)
        assert len(any_monitor.name) > 0

    def test_monitor_enabled_is_bool(self, any_monitor):
        assert isinstance(any_monitor.enabled, bool)

    def test_monitor_dimensions_positive(self, any_monitor):
        assert any_monitor.width > 0
        assert any_monitor.height > 0

    def test_monitor_by_id(self, zm_client, any_monitor):
        fetched = zm_client.monitor(any_monitor.id)
        assert fetched.id == any_monitor.id
        assert fetched.name == any_monitor.name

    def test_monitors_caching(self, zm_client):
        first = zm_client.monitors()
        second = zm_client.monitors()
        assert first is second  # same cached object

    def test_monitors_force_reload(self, zm_client):
        cached = zm_client.monitors()
        reloaded = zm_client.monitors(force_reload=True)
        assert cached is not reloaded
        assert len(reloaded) == len(cached)

    def test_alarm_status(self, zm_client, any_monitor):
        """alarm_status() should return a dict without error."""
        result = zm_client.alarm_status(any_monitor.id)
        assert isinstance(result, dict)

    def test_daemon_status(self, zm_client, any_monitor):
        """daemon_status() should return a dict without error."""
        result = zm_client.daemon_status(any_monitor.id)
        assert isinstance(result, dict)
