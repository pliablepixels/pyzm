"""E2E write tests for event notes (requires ZM_E2E_WRITE=1)."""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.zm_e2e, pytest.mark.zm_e2e_write]


class TestWriteNotes:
    def test_update_notes_roundtrip(self, zm_client, note_restorer):
        """Write notes, re-fetch, verify, then teardown restores original."""
        event = note_restorer
        marker = f"pyzm-e2e-{uuid.uuid4().hex[:8]}"
        event.update_notes(marker)

        fetched = zm_client.event(event.id)
        assert fetched.notes == marker

    def test_clear_notes(self, zm_client, note_restorer):
        """Setting notes to empty string clears them."""
        event = note_restorer
        event.update_notes("temporary")
        event.update_notes("")

        fetched = zm_client.event(event.id)
        assert fetched.notes == ""

    def test_unicode_notes(self, zm_client, note_restorer):
        """Non-ASCII text in notes is preserved."""
        event = note_restorer
        # Use BMP characters only — ZM's DB may not support 4-byte emoji
        text = "d\u00e9tected: person \u2714 caf\u00e9"
        event.update_notes(text)

        fetched = zm_client.event(event.id)
        assert fetched.notes == text
