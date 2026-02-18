"""E2E tests for ZoneMinder event operations."""

from __future__ import annotations

from datetime import datetime

import pytest

from pyzm.models.zm import Event

pytestmark = pytest.mark.zm_e2e


class TestEvents:
    def test_events_returns_list(self, zm_client):
        events = zm_client.events(limit=5)
        assert isinstance(events, list)

    def test_events_respects_limit(self, zm_client):
        events = zm_client.events(limit=3)
        assert len(events) <= 3

    def test_event_is_correct_type(self, any_event):
        assert isinstance(any_event, Event)

    def test_event_id_is_int(self, any_event):
        assert isinstance(any_event.id, int)
        assert any_event.id > 0

    def test_event_monitor_id_is_int(self, any_event):
        assert isinstance(any_event.monitor_id, int)
        assert any_event.monitor_id > 0

    def test_event_length_is_float(self, any_event):
        assert isinstance(any_event.length, float)

    def test_event_start_time_is_datetime(self, any_event):
        assert isinstance(any_event.start_time, datetime)

    def test_event_by_id(self, zm_client, any_event):
        fetched = zm_client.event(any_event.id)
        assert fetched.id == any_event.id

    def test_events_filter_by_monitor_id(self, zm_client, any_event):
        filtered = zm_client.events(monitor_id=any_event.monitor_id, limit=10)
        assert all(e.monitor_id == any_event.monitor_id for e in filtered)

    def test_events_filter_by_event_id(self, zm_client, any_event):
        filtered = zm_client.events(event_id=any_event.id)
        assert len(filtered) == 1
        assert filtered[0].id == any_event.id


class TestMonitorEventsOOP:
    """Tests for Monitor.events() OOP shim."""

    def test_monitor_events_oop(self, any_monitor):
        """Monitor.events() should return events scoped to that monitor."""
        events = any_monitor.events(limit=5)
        assert isinstance(events, list)
        for e in events:
            assert e.monitor_id == any_monitor.id


class TestEventFilters:
    """Tests for event filter parameters that use CakePHP path syntax."""

    def test_events_filter_since(self, zm_client, any_event):
        """since= should only return events at or after that time."""
        if any_event.start_time is None:
            pytest.skip("Event has no start_time")
        since_str = any_event.start_time.strftime("%Y-%m-%d %H:%M:%S")
        filtered = zm_client.events(since=since_str, limit=10)
        for e in filtered:
            assert e.start_time >= any_event.start_time

    def test_events_filter_until(self, zm_client, any_event):
        """until= should only return events at or before that time."""
        if any_event.start_time is None:
            pytest.skip("Event has no start_time")
        until_str = any_event.start_time.strftime("%Y-%m-%d %H:%M:%S")
        filtered = zm_client.events(until=until_str, limit=10)
        for e in filtered:
            assert e.start_time <= any_event.start_time

    def test_events_filter_min_alarm_frames(self, zm_client, any_event):
        """min_alarm_frames= should filter out low-alarm events."""
        threshold = max(1, any_event.alarm_frames)
        filtered = zm_client.events(min_alarm_frames=threshold, limit=10)
        for e in filtered:
            assert e.alarm_frames >= threshold

    def test_events_filter_object_only(self, zm_client, any_event):
        """object_only=True should only return events with 'detected' in Notes."""
        if "detected" not in (any_event.notes or "").lower():
            pytest.skip("No events with 'detected' in notes")
        filtered = zm_client.events(object_only=True, limit=10)
        for e in filtered:
            assert "detected" in (e.notes or "").lower()

    def test_events_compound_filter(self, zm_client, any_event):
        """Multiple filters combined should all apply."""
        filtered = zm_client.events(
            monitor_id=any_event.monitor_id,
            min_alarm_frames=1,
            limit=5,
        )
        for e in filtered:
            assert e.monitor_id == any_event.monitor_id
            assert e.alarm_frames >= 1
