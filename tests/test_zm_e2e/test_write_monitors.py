"""E2E write tests for monitor arm/disarm/update (requires ZM_E2E_WRITE=1)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zm_e2e, pytest.mark.zm_e2e_write]


class TestWriteMonitors:
    def test_arm_disarm_roundtrip(self, any_monitor):
        """arm() then disarm() should complete without error."""
        any_monitor.arm()
        any_monitor.disarm()

    def test_update_monitor_function(self, zm_client, any_monitor):
        """update() should change Function and revert it."""
        original = any_monitor.function
        # Set to Monitor (passive) then back
        target = "Monitor" if original != "Monitor" else "Modect"
        any_monitor.update(Function=target)

        # Re-fetch and verify
        updated = zm_client.monitor(any_monitor.id)
        assert updated.function == target

        # Revert
        any_monitor.update(Function=original)
        reverted = zm_client.monitor(any_monitor.id)
        assert reverted.function == original
