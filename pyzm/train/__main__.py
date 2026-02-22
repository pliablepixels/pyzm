"""CLI entry point: ``python -m pyzm.train``."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pyzm YOLO Fine-Tuning — headless CLI or Streamlit UI",
        prog="python -m pyzm.train",
    )

    # Positional: when given, run headless; when absent, launch Streamlit UI
    ap.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Path to YOLO dataset folder (headless mode). Omit to launch the UI.",
    )

    # --- Headless flags ---
    ap.add_argument(
        "--model",
        default="yolo11s",
        help="Base YOLO model (default: yolo11s)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (default: auto-detect from GPU)",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Train/val split ratio (default: 0.2)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="ONNX export path (default: auto in project dir)",
    )
    ap.add_argument(
        "--project-name",
        default=None,
        help="Project name (default: derived from dataset folder name)",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda:0, etc. (default: auto)",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Max images per class to import (default: 0 = all)",
    )

    # --- Shared flag ---
    ap.add_argument(
        "--workspace-dir",
        default=None,
        help="Training workspace (default: ~/.pyzm/training)",
    )

    # --- UI-only flags ---
    ap.add_argument(
        "--base-path",
        default="/var/lib/zmeventnotification/models",
        help="Model base path for auto-labeling (default: /var/lib/zmeventnotification/models)",
    )
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for UI (default: 0.0.0.0)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for UI (default: 8501)",
    )
    ap.add_argument(
        "--processor",
        default="gpu",
        choices=["cpu", "gpu"],
        help="Processor for auto-labeling (default: gpu)",
    )

    args = ap.parse_args()

    if args.dataset:
        _run_headless(args)
    else:
        _run_ui(args)


def _run_headless(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pyzm.train.pipeline import run_pipeline

    output = Path(args.output) if args.output else None
    workspace = Path(args.workspace_dir) if args.workspace_dir else None

    try:
        run_pipeline(
            Path(args.dataset),
            project_name=args.project_name,
            workspace_dir=workspace,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            val_ratio=args.val_ratio,
            output=output,
            device=args.device,
            max_per_class=args.max_per_class,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_ui(args: argparse.Namespace) -> None:
    from pyzm.train import check_dependencies
    check_dependencies()

    import subprocess

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        _app_path(),
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--theme.base", "dark",
        "--theme.primaryColor", "#a855f7",
        "--theme.backgroundColor", "#0d1117",
        "--theme.secondaryBackgroundColor", "#161b22",
        "--theme.textColor", "#f0f6fc",
        "--theme.font", "sans serif",
        "--",
        "--base-path", args.base_path,
        "--processor", args.processor,
    ]
    if args.workspace_dir:
        cmd.extend(["--workspace-dir", args.workspace_dir])

    sys.exit(subprocess.call(cmd))


def _app_path() -> str:
    """Return the absolute path to the Streamlit app module."""
    from pathlib import Path

    return str(Path(__file__).with_name("app.py"))


if __name__ == "__main__":
    main()
