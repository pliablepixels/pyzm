"""Tests for pyzm.train.__main__ CLI entry point."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from pyzm.train.__main__ import main, _run_headless, _run_ui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Call main() with *argv* and capture the parsed Namespace.

    Mocks both _run_headless and _run_ui so no real work is done.
    """
    with patch("sys.argv", ["python -m pyzm.train"] + argv):
        with patch("pyzm.train.__main__._run_headless") as mock_hl:
            with patch("pyzm.train.__main__._run_ui") as mock_ui:
                main()
                if mock_hl.called:
                    return mock_hl.call_args[0][0]
                return mock_ui.call_args[0][0]


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    def test_defaults_no_dataset(self):
        """No dataset arg -> UI mode with default values."""
        args = _parse_args([])
        assert args.dataset is None
        assert args.model == "yolo11s"
        assert args.epochs == 50
        assert args.batch is None
        assert args.imgsz == 640
        assert args.val_ratio == pytest.approx(0.2)
        assert args.output is None
        assert args.device == "auto"
        assert args.host == "0.0.0.0"
        assert args.port == 8501
        assert args.processor == "gpu"

    def test_custom_headless_values(self):
        """Custom CLI values are parsed correctly."""
        args = _parse_args([
            "/my/dataset",
            "--model", "yolo11m",
            "--epochs", "100",
            "--batch", "32",
            "--imgsz", "1280",
            "--val-ratio", "0.3",
            "--output", "/tmp/model.onnx",
            "--project-name", "my_project",
            "--device", "cuda:0",
            "--workspace-dir", "/tmp/ws",
        ])
        assert args.dataset == "/my/dataset"
        assert args.model == "yolo11m"
        assert args.epochs == 100
        assert args.batch == 32
        assert args.imgsz == 1280
        assert args.val_ratio == pytest.approx(0.3)
        assert args.output == "/tmp/model.onnx"
        assert args.project_name == "my_project"
        assert args.device == "cuda:0"
        assert args.workspace_dir == "/tmp/ws"

    def test_ui_flags(self):
        """UI-specific flags are parsed correctly."""
        args = _parse_args([
            "--host", "127.0.0.1",
            "--port", "9000",
            "--processor", "cpu",
            "--base-path", "/custom/path",
        ])
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.processor == "cpu"
        assert args.base_path == "/custom/path"


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

class TestModeDispatch:
    def test_dataset_triggers_headless(self):
        """Providing dataset arg triggers headless mode."""
        with patch("sys.argv", ["prog", "/path/to/dataset"]):
            with patch("pyzm.train.__main__._run_headless") as mock_hl:
                with patch("pyzm.train.__main__._run_ui") as mock_ui:
                    main()

        mock_hl.assert_called_once()
        mock_ui.assert_not_called()
        assert mock_hl.call_args[0][0].dataset == "/path/to/dataset"

    def test_no_dataset_triggers_ui(self):
        """No dataset arg triggers UI mode."""
        with patch("sys.argv", ["prog"]):
            with patch("pyzm.train.__main__._run_headless") as mock_hl:
                with patch("pyzm.train.__main__._run_ui") as mock_ui:
                    main()

        mock_ui.assert_called_once()
        mock_hl.assert_not_called()


# ---------------------------------------------------------------------------
# _run_headless
# ---------------------------------------------------------------------------

class TestRunHeadless:
    def test_calls_run_pipeline(self):
        """_run_headless forwards parsed args to run_pipeline."""
        args = argparse.Namespace(
            dataset="/path/to/dataset",
            model="yolo11s",
            epochs=50,
            batch=None,
            imgsz=640,
            val_ratio=0.2,
            output=None,
            project_name=None,
            device="auto",
            workspace_dir=None,
            max_per_class=0,
            mode="new_class",
        )

        with patch("pyzm.train.pipeline.run_pipeline") as mock_rp:
            _run_headless(args)

        mock_rp.assert_called_once()
        call_kwargs = mock_rp.call_args
        # First positional arg is the dataset Path
        assert str(call_kwargs[0][0]) == "/path/to/dataset"
        assert call_kwargs[1]["epochs"] == 50
        assert call_kwargs[1]["batch"] is None
        assert call_kwargs[1]["mode"] == "new_class"

    def test_pipeline_error_exits(self):
        """_run_headless exits with code 1 on pipeline ValueError."""
        args = argparse.Namespace(
            dataset="/nonexistent",
            model="yolo11s",
            epochs=50,
            batch=None,
            imgsz=640,
            val_ratio=0.2,
            output=None,
            project_name=None,
            device="auto",
            workspace_dir=None,
            max_per_class=0,
            mode="new_class",
        )

        with patch("pyzm.train.pipeline.run_pipeline", side_effect=ValueError("bad")):
            with pytest.raises(SystemExit) as exc_info:
                _run_headless(args)
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _run_ui
# ---------------------------------------------------------------------------

class TestRunUI:
    def test_calls_streamlit_subprocess(self):
        """UI mode runs streamlit via subprocess.call."""
        args = argparse.Namespace(
            host="0.0.0.0",
            port=8501,
            base_path="/var/lib/zmeventnotification/models",
            processor="gpu",
            workspace_dir=None,
        )

        with patch("pyzm.train.check_dependencies"):
            with patch("subprocess.call", return_value=0) as mock_call:
                with pytest.raises(SystemExit) as exc_info:
                    _run_ui(args)
                assert exc_info.value.code == 0

        mock_call.assert_called_once()
        cmd = mock_call.call_args[0][0]
        # Should invoke streamlit
        assert cmd[1] == "-m"
        assert cmd[2] == "streamlit"
        assert "run" in cmd
        # Server address and port
        assert "--server.address" in cmd
        assert "0.0.0.0" in cmd
        assert "--server.port" in cmd
        assert "8501" in cmd

    def test_passes_workspace_dir(self):
        """When workspace_dir is set, it's forwarded to the streamlit command."""
        args = argparse.Namespace(
            host="0.0.0.0",
            port=8501,
            base_path="/models",
            processor="gpu",
            workspace_dir="/custom/ws",
        )

        with patch("pyzm.train.check_dependencies"):
            with patch("subprocess.call", return_value=0) as mock_call:
                with pytest.raises(SystemExit):
                    _run_ui(args)

        cmd = mock_call.call_args[0][0]
        assert "--workspace-dir" in cmd
        assert "/custom/ws" in cmd
