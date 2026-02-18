"""E2E tests for ZM states listing (readonly)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.zm_e2e


class TestStates:
    def test_states_list(self, zm_client):
        """states() should return a list of state dicts."""
        result = zm_client.states()
        assert isinstance(result, list)
        # ZM always has at least the 'default' state
        if result:
            assert "Name" in result[0]
