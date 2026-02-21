"""Tests for merged backend functionality.

Covers:
  - PortalockerMixin acquire/release lifecycle
  - create_yolo_backend() factory dispatch
  - ALPR bug fixes (url, filename, options references)
  - ALPR security fix (no shell=True in subprocess)

Refs #23
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, mock_open

import pytest

from pyzm.models.config import ModelConfig, ModelFramework, Processor


# ===================================================================
# PortalockerMixin
# ===================================================================


class TestPortalockerMixin:
    """Tests for PortalockerMixin acquire/release lifecycle."""

    def _make_mixin(self, *, disable_locks: bool = False):
        from pyzm.ml.backends.base import PortalockerMixin

        class Stub(PortalockerMixin):
            name = "test_stub"

        obj = Stub()
        obj._config = ModelConfig(
            processor=Processor.CPU,
            disable_locks=disable_locks,
            max_processes=2,
            max_lock_wait=60,
        )
        return obj

    @patch("pyzm.ml.backends.base.portalocker")
    def test_init_lock_creates_semaphore(self, mock_portalocker):
        mixin = self._make_mixin()
        mixin._init_lock()

        mock_portalocker.BoundedSemaphore.assert_called_once_with(
            maximum=2, name=mixin._lock_name, timeout=60
        )
        assert mixin._is_locked is False
        assert mixin._disable_locks is False

    def test_init_lock_disabled_skips_semaphore(self):
        mixin = self._make_mixin(disable_locks=True)
        mixin._init_lock()
        assert mixin._disable_locks is True
        assert not hasattr(mixin, "_lock")

    @patch("pyzm.ml.backends.base.portalocker")
    def test_acquire_release_cycle(self, mock_portalocker):
        mixin = self._make_mixin()
        mixin._init_lock()

        mixin.acquire_lock()
        assert mixin._is_locked is True
        mixin._lock.acquire.assert_called_once()

        mixin.release_lock()
        assert mixin._is_locked is False
        mixin._lock.release.assert_called_once()

    @patch("pyzm.ml.backends.base.portalocker")
    def test_double_acquire_is_noop(self, mock_portalocker):
        mixin = self._make_mixin()
        mixin._init_lock()

        mixin.acquire_lock()
        mixin.acquire_lock()  # second call should be no-op
        assert mixin._lock.acquire.call_count == 1

    @patch("pyzm.ml.backends.base.portalocker")
    def test_double_release_is_noop(self, mock_portalocker):
        mixin = self._make_mixin()
        mixin._init_lock()

        mixin.acquire_lock()
        mixin.release_lock()
        mixin.release_lock()  # second call should be no-op
        assert mixin._lock.release.call_count == 1

    def test_acquire_disabled_is_noop(self):
        mixin = self._make_mixin(disable_locks=True)
        mixin._init_lock()

        mixin.acquire_lock()  # should not raise
        assert mixin._is_locked is False

    def test_release_disabled_is_noop(self):
        mixin = self._make_mixin(disable_locks=True)
        mixin._init_lock()

        mixin.release_lock()  # should not raise

    def test_acquire_timeout_raises_valueerror(self):
        import portalocker

        mixin = self._make_mixin()
        mixin._init_lock()
        # Replace the real lock with a mock that raises AlreadyLocked
        mock_lock = MagicMock()
        mock_lock.acquire.side_effect = portalocker.AlreadyLocked
        mixin._lock = mock_lock

        with pytest.raises(ValueError, match="timeout"):
            mixin.acquire_lock()


# ===================================================================
# create_yolo_backend factory
# ===================================================================


class TestCreateYoloBackend:
    """Tests for create_yolo_backend() dispatch."""

    @patch("pyzm.ml.backends.yolo_onnx.YoloOnnx.__init__", return_value=None)
    def test_onnx_extension_returns_yolo_onnx(self, mock_init):
        from pyzm.ml.backends.yolo import create_yolo_backend
        from pyzm.ml.backends.yolo_onnx import YoloOnnx

        config = ModelConfig(weights="/path/to/model.onnx")
        backend = create_yolo_backend(config)
        assert isinstance(backend, YoloOnnx)

    @patch("pyzm.ml.backends.yolo_darknet.YoloDarknet.__init__", return_value=None)
    def test_weights_extension_returns_yolo_darknet(self, mock_init):
        from pyzm.ml.backends.yolo import create_yolo_backend
        from pyzm.ml.backends.yolo_darknet import YoloDarknet

        config = ModelConfig(weights="/path/to/model.weights")
        backend = create_yolo_backend(config)
        assert isinstance(backend, YoloDarknet)

    @patch("pyzm.ml.backends.yolo_darknet.YoloDarknet.__init__", return_value=None)
    def test_no_weights_returns_darknet(self, mock_init):
        from pyzm.ml.backends.yolo import create_yolo_backend
        from pyzm.ml.backends.yolo_darknet import YoloDarknet

        config = ModelConfig()
        backend = create_yolo_backend(config)
        assert isinstance(backend, YoloDarknet)

    @patch("pyzm.ml.backends.yolo_onnx.YoloOnnx.__init__", return_value=None)
    def test_case_insensitive_onnx(self, mock_init):
        from pyzm.ml.backends.yolo import create_yolo_backend
        from pyzm.ml.backends.yolo_onnx import YoloOnnx

        config = ModelConfig(weights="/path/to/MODEL.ONNX")
        backend = create_yolo_backend(config)
        assert isinstance(backend, YoloOnnx)


# ===================================================================
# ALPR bug fixes
# ===================================================================


class TestAlprBugFixes:
    """Verify bug fixes in the merged ALPR backend."""

    def test_open_alpr_uses_self_url(self):
        """Bug fix: alpr.py:259 — url → self.url."""
        from pyzm.ml.backends.alpr import _OpenAlpr

        config = ModelConfig(
            alpr_url="https://custom.api/v2",
            alpr_key="test-key",
            alpr_service="open_alpr",
        )
        service = _OpenAlpr(config)
        assert service.url == "https://custom.api/v2"

    def test_open_alpr_default_url(self):
        """When no url provided, should use default."""
        from pyzm.ml.backends.alpr import _OpenAlpr

        config = ModelConfig(alpr_service="open_alpr")
        service = _OpenAlpr(config)
        assert service.url == "https://api.openalpr.com/v2/recognize"

    def test_open_alpr_cmdline_no_shell_true(self):
        """Security fix: subprocess should not use shell=True."""
        from pyzm.ml.backends.alpr import _OpenAlprCmdLine

        config = ModelConfig(
            alpr_service="open_alpr_cmdline",
            options={
                "openalpr_cmdline_binary": "/usr/bin/alpr",
                "openalpr_cmdline_params": "-c us",
                "openalpr_cmdline_min_confidence": 0.3,
            },
        )
        service = _OpenAlprCmdLine(config)
        # Base cmd should contain -j for JSON output
        assert "-j" in service._base_cmd

    @patch("subprocess.run")
    @patch("cv2.imwrite")
    def test_cmdline_uses_subprocess_run(self, mock_imwrite, mock_run):
        """Verify subprocess.run is called instead of check_output with shell=True."""
        import numpy as np
        from pyzm.ml.backends.alpr import _OpenAlprCmdLine

        config = ModelConfig(
            alpr_service="open_alpr_cmdline",
            options={
                "openalpr_cmdline_binary": "/usr/bin/alpr",
                "openalpr_cmdline_params": "-c us",
                "openalpr_cmdline_min_confidence": 0.3,
            },
        )
        service = _OpenAlprCmdLine(config)

        mock_run.return_value = MagicMock(stdout='{"results": []}', returncode=0)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service.detect(dummy_image, "test_alpr")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        # Should be a list, not a string
        assert isinstance(call_args[0][0], list)
        # Should NOT have shell=True
        assert call_args.kwargs.get("shell") is not True

    def test_plate_recognizer_uses_config_fields(self):
        """Bug fix: options.get → self._config field access."""
        from pyzm.ml.backends.alpr import _PlateRecognizer

        config = ModelConfig(
            alpr_url="https://api.platerecognizer.com/v1",
            alpr_key="my-key",
            alpr_service="plate_recognizer",
            platerec_min_dscore=0.5,
            platerec_min_score=0.7,
        )
        service = _PlateRecognizer(config)
        assert service._config.platerec_min_dscore == 0.5
        assert service._config.platerec_min_score == 0.7


# ===================================================================
# Pipeline factory with merged backends
# ===================================================================


class TestPipelineFactory:
    """Verify _create_backend returns merged backend classes."""

    @patch("pyzm.ml.backends.yolo.PortalockerMixin._init_lock")
    def test_opencv_creates_yolo_via_factory(self, mock_lock):
        from pyzm.ml.pipeline import _create_backend
        from pyzm.ml.backends.yolo import YoloBase

        config = ModelConfig(
            framework=ModelFramework.OPENCV,
            weights="/path/to/yolov4.weights",
        )
        backend = _create_backend(config)
        assert isinstance(backend, YoloBase)

    @patch("pyzm.ml.backends.yolo.PortalockerMixin._init_lock")
    def test_opencv_onnx_creates_yolo_onnx(self, mock_lock):
        from pyzm.ml.pipeline import _create_backend
        from pyzm.ml.backends.yolo_onnx import YoloOnnx

        config = ModelConfig(
            framework=ModelFramework.OPENCV,
            weights="/path/to/yolo11s.onnx",
        )
        backend = _create_backend(config)
        assert isinstance(backend, YoloOnnx)

    def test_face_dlib_creates_face_dlib_backend(self):
        from pyzm.ml.pipeline import _create_backend
        from pyzm.ml.backends.face_dlib import FaceDlibBackend

        config = ModelConfig(
            framework=ModelFramework.FACE_DLIB,
            known_faces_dir="/tmp/faces",
            disable_locks=True,
        )
        backend = _create_backend(config)
        assert isinstance(backend, FaceDlibBackend)

    def test_face_tpu_creates_face_tpu_backend(self):
        from pyzm.ml.pipeline import _create_backend
        from pyzm.ml.backends.face_tpu import FaceTpuBackend

        config = ModelConfig(
            framework=ModelFramework.FACE_TPU,
            disable_locks=True,
        )
        backend = _create_backend(config)
        assert isinstance(backend, FaceTpuBackend)
