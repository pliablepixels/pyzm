"""pyzm.train -- YOLO fine-tuning training module.

Provides a guided Streamlit web UI **or** headless CLI for fine-tuning YOLO
models on custom images so ZoneMinder users can detect custom objects
alongside COCO classes.

Install::

    pip install pyzm[train]

Run (UI)::

    python -m pyzm.train

Run (headless)::

    python -m pyzm.train /path/to/yolo-dataset

Programmatic::

    from pyzm.train import run_pipeline
    result = run_pipeline(Path("/path/to/yolo-dataset"), epochs=50)
"""

from __future__ import annotations

from pyzm.train.pipeline import run_pipeline

__all__: list[str] = ["run_pipeline"]


def check_dependencies() -> None:
    """Raise ImportError if required extras are missing."""
    missing: list[str] = []

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        missing.append("ultralytics>=8.3")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        missing.append("streamlit>=1.41")

    try:
        import streamlit_drawable_canvas  # noqa: F401
    except ImportError:
        missing.append("streamlit-drawable-canvas>=0.9")

    try:
        import st_clickable_images  # noqa: F401
    except ImportError:
        missing.append("st-clickable-images>=0.0.3")

    if missing:
        raise ImportError(
            "pyzm.train requires extra dependencies. Install with:\n\n"
            "  pip install pyzm[train]\n\n"
            f"Missing: {', '.join(missing)}"
        )
