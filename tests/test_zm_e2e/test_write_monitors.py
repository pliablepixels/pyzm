"""E2E write tests for monitor arm/disarm/update (requires ZM_E2E_WRITE=1)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zm_e2e, pytest.mark.zm_e2e_write]


class TestWriteMonitors:
    def test_arm_disarm_roundtrip(self, zm_client, any_monitor):
        """arm() then disarm() should complete without error."""
        zm_client.arm(any_monitor.id)
        zm_client.disarm(any_monitor.id)

    def test_update_monitor_function(self, zm_client, any_monitor):
        """update_monitor() should change Function and revert it."""
        original = any_monitor.function
        # Set to Monitor (passive) then back
        target = "Monitor" if original != "Monitor" else "Modect"
        zm_client.update_monitor(any_monitor.id, Function=target)

        # Re-fetch and verify
        updated = zm_client.monitor(any_monitor.id)
        assert updated.function == target

        # Revert
        zm_client.update_monitor(any_monitor.id, Function=original)
        reverted = zm_client.monitor(any_monitor.id)
        assert reverted.function == original
