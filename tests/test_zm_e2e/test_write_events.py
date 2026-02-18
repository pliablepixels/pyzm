"""E2E write tests for event deletion (requires ZM_E2E_WRITE=1).

NOTE: Do NOT use ZM_E2E_WRITE env var in test logic -- the user will
run these manually with ``ZM_E2E_WRITE=1 pytest tests/test_zm_e2e/ -v``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zm_e2e, pytest.mark.zm_e2e_write]


class TestWriteEvents:
    def test_delete_event(self, zm_client):
        """delete_event() should delete an event. Picks the oldest event."""
        events = zm_client.events(limit=1)
        if not events:
            pytest.skip("No events on ZM server to delete")

        ev = events[0]
        zm_client.delete_event(ev.id)

        # Verify it's gone
        with pytest.raises(ValueError, match="not found"):
            zm_client.event(ev.id)
