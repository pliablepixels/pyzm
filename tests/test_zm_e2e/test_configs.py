"""E2E tests for ZM configuration endpoints (readonly)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.zm_e2e


class TestConfigs:
    def test_configs_list(self, zm_client):
        """configs() should return a non-empty list of config dicts."""
        cfgs = zm_client.configs()
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0
        assert "Name" in cfgs[0]

    def test_config_by_name(self, zm_client):
        """config() should fetch a known config parameter."""
        cfg = zm_client.config("ZM_LANG_DEFAULT")
        assert isinstance(cfg, dict)
        assert "Value" in cfg
        # Name is always present (injected by setdefault on newer ZM)
        assert cfg["Name"] == "ZM_LANG_DEFAULT"
