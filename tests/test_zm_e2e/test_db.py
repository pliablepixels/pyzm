"""E2E tests for DB-dependent ZMClient methods (tag_event, event_path).

These tests require the ZM database to be accessible (reads /etc/zm/zm.conf).
Skipped automatically when DB is unavailable (permissions, missing config, etc).
"""

from __future__ import annotations

import pytest

from pyzm.zm.db import get_zm_db
from tests.test_zm_e2e.conftest import zm_db_available

pytestmark = pytest.mark.zm_e2e

_db_ok, _db_reason = zm_db_available()
skip_no_db = pytest.mark.skipif(not _db_ok, reason=_db_reason or "ZM database not accessible")


@skip_no_db
class TestEventPath:
    def test_event_path_returns_string(self, any_event):
        """path() should return a filesystem path string."""
        path = any_event.path()
        assert path is not None, "path() returned None"
        assert isinstance(path, str)
        assert len(path) > 0

    def test_event_path_contains_monitor_id(self, any_event):
        """The path should include the monitor ID."""
        path = any_event.path()
        if path is None:
            pytest.skip("path() returned None")
        assert str(any_event.monitor_id) in path


@skip_no_db
@pytest.mark.zm_e2e_write
class TestTagEvent:
    def _cleanup_tags(self, event_id: int, labels: list[str]):
        """Remove test tags from the event (best-effort cleanup)."""
        conn = get_zm_db()
        if conn is None:
            return
        try:
            cur = conn.cursor(dictionary=True)
            for label in labels:
                cur.execute("SELECT Id FROM Tags WHERE Name=%s", (label,))
                row = cur.fetchone()
                if row:
                    tag_id = row["Id"]
                    cur.execute(
                        "DELETE FROM Events_Tags WHERE TagId=%s AND EventId=%s",
                        (tag_id, event_id),
                    )
                    cur.execute("DELETE FROM Tags WHERE Id=%s", (tag_id,))
            conn.commit()
            cur.close()
        finally:
            conn.close()

    def test_tag_event_creates_tag(self, any_event):
        """tag() should create tags and link them to the event."""
        labels = ["pyzm_e2e_test_label"]
        try:
            any_event.tag(labels)
            conn = get_zm_db()
            assert conn is not None
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT Id FROM Tags WHERE Name=%s", (labels[0],))
            row = cur.fetchone()
            assert row is not None, "Tag was not created in DB"
            tag_id = row["Id"]
            cur.execute(
                "SELECT * FROM Events_Tags WHERE TagId=%s AND EventId=%s",
                (tag_id, any_event.id),
            )
            link = cur.fetchone()
            assert link is not None, "Tag was not linked to event"
            cur.close()
            conn.close()
        finally:
            self._cleanup_tags(any_event.id, labels)

    def test_tag_event_deduplicates(self, any_event):
        """Passing duplicate labels should not create duplicate tags."""
        labels = ["pyzm_e2e_dedup", "pyzm_e2e_dedup"]
        try:
            any_event.tag(labels)
            conn = get_zm_db()
            assert conn is not None
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT COUNT(*) as cnt FROM Tags WHERE Name=%s", ("pyzm_e2e_dedup",))
            row = cur.fetchone()
            assert row["cnt"] == 1
            cur.close()
            conn.close()
        finally:
            self._cleanup_tags(any_event.id, ["pyzm_e2e_dedup"])

    def test_tag_event_empty_list_is_noop(self, any_event):
        """Passing an empty list should not raise."""
        any_event.tag([])
