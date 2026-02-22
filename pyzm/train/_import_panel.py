"""Phase 1: Select Images -- import frames from ZM events, YOLO datasets, or raw images."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import streamlit as st

from pyzm.train.app import MIN_IMAGES_PER_CLASS, _section_header
from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")


# ===================================================================
# Auto-detect
# ===================================================================

def _auto_detect_image(
    image_path: Path,
    args: argparse.Namespace,
) -> list[VerifiedDetection]:
    """Run auto-detect on a single image and return PENDING VerifiedDetections."""
    base_model = st.session_state.get("base_model", "yolo11s")
    model_classes = st.session_state.get("model_class_names", [])
    pdir = st.session_state.get("workspace_dir")
    best_pt = Path(pdir) / "runs" / "train" / "weights" / "best.pt" if pdir else None
    has_trained = best_pt is not None and best_pt.exists()

    detections: list[VerifiedDetection] = []
    if not (model_classes or has_trained):
        return detections

    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return detections
        h, w = img.shape[:2]
        from pyzm.ml.detector import Detector
        model_to_use = str(best_pt) if has_trained else base_model
        det = Detector(
            models=[model_to_use],
            base_path=args.base_path,
            processor=args.processor,
        )
        result = det.detect(img)
        for j, d in enumerate(result.detections):
            b = d.bbox
            cx = ((b.x1 + b.x2) / 2) / w
            cy = ((b.y1 + b.y2) / 2) / h
            bw = (b.x2 - b.x1) / w
            bh = (b.y2 - b.y1) / h
            ann = Annotation(class_id=0, cx=cx, cy=cy, w=bw, h=bh)
            detections.append(VerifiedDetection(
                detection_id=f"det_{j}",
                original=ann,
                status=DetectionStatus.PENDING,
                original_label=d.label,
                confidence=getattr(d, "confidence", None),
            ))
    except Exception as exc:
        logger.warning("Auto-detect failed for %s: %s", image_path.name, exc)

    return detections


# ===================================================================
# Upload panel
# ===================================================================

def _upload_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
    *,
    target_classes: list[str] | None = None,
    label: str = "Upload images where detection failed or needs improvement",
) -> None:
    if target_classes:
        st.caption(f"Upload images containing: **{', '.join(target_classes)}**")
    upload_key = st.session_state.get("_upload_key", 0)
    uploaded = st.file_uploader(
        label,
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{upload_key}",
    )
    if not uploaded:
        return

    # Phase 1: save all images to disk
    import_bar = st.progress(0, text="Importing images...")
    destinations: list[Path] = []
    for i, f in enumerate(uploaded):
        tmp = Path(tempfile.mkdtemp()) / f.name
        tmp.write_bytes(f.read())
        destinations.append(ds.add_image(tmp, []))
        import_bar.progress((i + 1) / len(uploaded), text=f"Importing {i + 1}/{len(uploaded)}")
    import_bar.empty()

    # Create empty verification entries (detection deferred to review phase)
    for dest in destinations:
        store.set(ImageVerification(
            image_name=dest.name,
            detections=[],
            fully_reviewed=False,
        ))

    store.save()
    st.session_state["_upload_key"] = upload_key + 1
    st.toast(f"Added {len(destinations)} images")
    st.rerun()


# ===================================================================
# PHASE 1: Select Images
# ===================================================================

def _phase_select(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    _section_header("&#x1F4F7;", "Select Images")

    # Show banner when classes need more training images
    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
    if needs:
        summary = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in needs
        )
        st.info(f"Classes needing more images: {summary}")

    source = st.radio(
        "Data source",
        ["Pre-Annotated YOLO Dataset", "Raw Images", "ZoneMinder Events"],
        horizontal=True,
        key="data_source",
    )

    if source == "Pre-Annotated YOLO Dataset":
        st.caption("Import a pre-annotated dataset in YOLO format.")
        from pyzm.train.local_import import local_dataset_panel
        local_dataset_panel(ds, store, args)
    elif source == "Raw Images":
        st.caption("Import unannotated images for manual annotation.")
        from pyzm.train.local_import import raw_images_panel
        raw_images_panel(ds, store, args)
    else:
        st.caption("Select events where detection was wrong or missing.")
        from pyzm.train.zm_browser import zm_event_browser_panel
        zm_event_browser_panel(ds, store, args)

    images = ds.staged_images()
    if images:
        st.divider()
        st.success(f"{len(images)} image{'s' if len(images) != 1 else ''} imported. Ready for review.")
        if st.button(":material/rate_review: Go to Review", type="primary"):
            st.session_state["active_phase"] = "review"
            st.session_state.pop("_auto_label", None)
            st.rerun()
        # Show image list in collapsed expander; cap to avoid slow renders
        max_show = 200
        with st.expander(f"Imported images ({len(images)})", expanded=False):
            for img in images[:max_show]:
                iv = store.get(img.name)
                status = "\u2713" if iv and iv.fully_reviewed else "\u23f3"
                det_count = len(iv.detections) if iv else 0
                st.caption(f"{status} {img.name} ({det_count} annotations)")
            if len(images) > max_show:
                st.caption(f"... and {len(images) - max_show} more")
