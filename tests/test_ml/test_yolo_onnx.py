"""Tests for YoloOnnx._parse_native_e2e garbled-output detection.

Covers three scenarios:
  1. Near-zero confidences  → falls back to pre-NMS layer
  2. Identical confidences  → falls back to pre-NMS layer
  3. Valid e2e output       → returns detections directly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo(*, pre_nms_layer: str | None = "pre_nms"):
    """Build a minimal YoloOnnx with the attrs _parse_native_e2e needs."""
    from pyzm.ml.backends.yolo_onnx import YoloOnnx
    from pyzm.models.config import ModelConfig

    config = ModelConfig(name="test_yolo", weights="test.onnx", disable_locks=True)
    obj = object.__new__(YoloOnnx)
    obj._config = config
    obj.net = MagicMock()
    obj.is_native_e2e = True
    obj.pre_nms_layer = pre_nms_layer
    obj._lb_scale = 1.0
    obj._lb_pad_w = 0
    obj._lb_pad_h = 0
    return obj


def _e2e_output(confs, *, n_cols=6):
    """Build a (N, 6) e2e output array with given confidence column."""
    n = len(confs)
    out = np.zeros((n, n_cols), dtype=np.float32)
    # x1, y1, x2, y2 — dummy boxes
    out[:, 0] = 10
    out[:, 1] = 10
    out[:, 2] = 50
    out[:, 3] = 50
    out[:, 4] = np.array(confs, dtype=np.float32)
    out[:, 5] = 0  # class_id
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseNativeE2E:
    """Unit tests for _parse_native_e2e garbled-output detection."""

    def test_near_zero_confs_falls_back(self):
        """All confidences < 0.01 → should fall back to pre-NMS layer."""
        yolo = _make_yolo()
        # 300 near-zero confidences (mimics garbled bird.jpg output)
        confs = np.random.uniform(0.0, 0.001, size=300).tolist()
        yolo.net.forward.return_value = _e2e_output(confs)

        fallback_result = ([0], [0.9755], [[10, 10, 40, 40]])

        blob = MagicMock()
        with patch.object(yolo, '_forward_and_parse', return_value=fallback_result) as mock_fwd:
            ids, scores, boxes = yolo._parse_native_e2e(blob, conf_threshold=0.3)

        mock_fwd.assert_called_once_with(blob, 0, 0, 0.3)
        assert yolo.is_native_e2e is False
        assert ids == [0]
        assert scores == [0.9755]

    def test_near_zero_confs_no_pre_nms(self):
        """Near-zero confs without pre_nms_layer → returns empty."""
        yolo = _make_yolo(pre_nms_layer=None)
        confs = np.random.uniform(0.0, 0.001, size=300).tolist()
        yolo.net.forward.return_value = _e2e_output(confs)

        ids, scores, boxes = yolo._parse_native_e2e(MagicMock(), conf_threshold=0.3)

        assert ids == []
        assert scores == []
        assert boxes == []

    def test_identical_confs_falls_back(self):
        """Many detections with identical confidence → falls back (existing check)."""
        yolo = _make_yolo()
        # 300 detections all with conf=0.1274 (garbled multiple.jpg pattern)
        confs = [0.1274] * 300
        yolo.net.forward.return_value = _e2e_output(confs)

        fallback_result = ([0, 1], [0.85, 0.72], [[10, 10, 40, 40], [60, 60, 90, 90]])

        blob = MagicMock()
        with patch.object(yolo, '_forward_and_parse', return_value=fallback_result) as mock_fwd:
            ids, scores, boxes = yolo._parse_native_e2e(blob, conf_threshold=0.3)

        mock_fwd.assert_called_once_with(blob, 0, 0, 0.3)
        assert yolo.is_native_e2e is False
        assert ids == [0, 1]

    def test_valid_output_returns_detections(self):
        """Valid e2e output with diverse confidences → returns detections directly."""
        yolo = _make_yolo()
        confs = [0.95, 0.87, 0.72, 0.55, 0.30]
        yolo.net.forward.return_value = _e2e_output(confs)

        ids, scores, boxes = yolo._parse_native_e2e(MagicMock(), conf_threshold=0.5)

        # Only confs >= 0.5 should remain: 0.95, 0.87, 0.72, 0.55
        assert len(ids) == 4
        assert scores == pytest.approx([0.95, 0.87, 0.72, 0.55], abs=1e-4)
        assert all(cid == 0 for cid in ids)

    def test_valid_output_no_fallback(self):
        """Valid output should NOT trigger fallback — _forward_and_parse not called."""
        yolo = _make_yolo()
        confs = [0.90, 0.80, 0.70]
        yolo.net.forward.return_value = _e2e_output(confs)

        with patch.object(yolo, '_forward_and_parse') as mock_fwd:
            yolo._parse_native_e2e(MagicMock(), conf_threshold=0.3)

        mock_fwd.assert_not_called()
        assert yolo.is_native_e2e is True
