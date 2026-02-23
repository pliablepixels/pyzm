"""Phase 1: Select Images -- import from YOLO datasets, raw images, or correct model detections."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import streamlit as st

from pyzm.train.app import (
    MIN_IMAGES_PER_CLASS,
    _read_model_classes,
    _scan_models,
    _section_header,
    _step_expander,
)
from pyzm.train.dataset import YOLODataset
from pyzm.train.verification import (
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")


# ===================================================================
# Auto-detect (thin wrapper reading st.session_state)
# ===================================================================

def _auto_detect_image(
    image_path: Path,
    args: argparse.Namespace,
) -> list[VerifiedDetection]:
    """Run auto-detect on a single image and return PENDING VerifiedDetections.

    Thin wrapper that reads ``st.session_state`` and delegates to
    :func:`pyzm.train.local_import.auto_detect_image`.
    """
    from pyzm.train.local_import import auto_detect_image

    return auto_detect_image(
        image_path,
        base_model=st.session_state.get("base_model", "yolo11s"),
        workspace_dir=st.session_state.get("workspace_dir"),
        base_path=args.base_path,
        processor=args.processor,
        model_classes=st.session_state.get("model_class_names", []),
    )


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
    data_source: str | None = None,
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
    if data_source:
        ds.set_setting("data_source", data_source)
    st.session_state["_upload_key"] = upload_key + 1
    st.toast(f"Added {len(destinations)} images")
    st.rerun()


# ===================================================================
# PHASE 1: Select Images
# ===================================================================

def _data_source_radio(ds: YOLODataset) -> str:
    """Render data source radio, restoring previous selection from project settings."""
    _SOURCE_OPTIONS = [
        "Manually annotate",
        "Import pre-annotated dataset",
    ]
    _SETTING_TO_INDEX = {
        "raw_images": 0,
        "correct_model": 0,
        "yolo_dataset": 1,
    }
    saved = ds.get_setting("data_source", "")
    index = _SETTING_TO_INDEX.get(saved, 0)
    return st.radio(
        "How do you want to add images?",
        _SOURCE_OPTIONS,
        horizontal=True,
        index=index,
        key="data_source",
    )


def _data_source_panels(
    source: str,
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
) -> None:
    """Render the import panel for the selected data source."""
    if source == "Manually annotate":
        st.caption(
            "Upload or point at a folder of images. The model can auto-detect "
            "objects — you then correct mistakes, draw missing boxes, and "
            "delete false detections in the Review phase."
        )
        from pyzm.train.local_import import manual_annotate_panel
        manual_annotate_panel(ds, store, args)
    else:
        st.caption(
            "Import a pre-annotated dataset in YOLO format (e.g. from Roboflow). "
            "Annotations are imported as-is."
        )
        from pyzm.train.local_import import local_dataset_panel
        local_dataset_panel(ds, store, args)


def _images_management(
    ds: YOLODataset,
    store: VerificationStore,
    images: list[Path],
) -> None:
    """Show imported images summary with per-image remove and Remove All."""
    from pyzm.train.app import _invalidate_thumbnail

    reviewed = store.reviewed_images_count()
    pending = len(images) - reviewed
    st.success(
        f"**{len(images)}** images imported "
        f"({reviewed} reviewed, {pending} pending)"
    )

    # Scrollable list with per-image remove buttons
    with st.container(height=250):
        for img_path in images:
            col_name, col_btn = st.columns([5, 1])
            with col_name:
                iv = store.get(img_path.name)
                icon = ":material/check_circle:" if iv and iv.fully_reviewed else ":material/pending:"
                st.markdown(f"{icon} {img_path.name}")
            with col_btn:
                if st.button(
                    ":material/close:",
                    key=f"rm_{img_path.name}",
                    help=f"Remove {img_path.name}",
                ):
                    ds.remove_image(img_path.name)
                    store.remove(img_path.name)
                    store.save()
                    _invalidate_thumbnail(img_path.name)
                    st.toast(f"Removed {img_path.name}")
                    st.rerun()

    # Remove All button with confirmation
    if st.button(":material/delete_sweep: Remove All Images", key="remove_all_images"):
        st.session_state["_confirm_remove_all"] = True
        st.rerun()

    if st.session_state.get("_confirm_remove_all"):
        st.warning(f"Remove all **{len(images)}** images and their annotations?")
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("Yes, remove all", type="primary", key="remove_all_yes"):
                for img_path in images:
                    ds.remove_image(img_path.name)
                    _invalidate_thumbnail(img_path.name)
                store.clear_all()
                store.save()
                st.session_state.pop("_confirm_remove_all", None)
                st.session_state["_thumb_cache"] = {}
                st.toast(f"Removed {len(images)} images")
                st.rerun()
        with no_col:
            if st.button("Cancel", key="remove_all_no"):
                st.session_state.pop("_confirm_remove_all", None)
                st.rerun()


def _model_step(ds: YOLODataset, args: argparse.Namespace) -> None:
    """Step expander for base model selection inside the Select phase."""
    import json

    has_model = bool(st.session_state.get("base_model"))
    model_detail = st.session_state.get("base_model", "") if has_model else ""
    with _step_expander("Base model", done=has_model, detail=model_detail):
        st.caption(
            "Pick the model you currently use for detection. "
            "We'll improve it using your corrections."
        )
        available = _scan_models(args.base_path)
        model_names = [m["name"] for m in available]
        model_paths = {m["name"]: m["path"] for m in available}

        default_idx = 0
        saved_model = st.session_state.get("base_model")
        for i, name in enumerate(model_names):
            if saved_model and name == saved_model:
                default_idx = i
                break
            if name == "yolo11s":
                default_idx = i

        base_model = st.selectbox(
            "Base model",
            options=model_names,
            index=default_idx,
            format_func=lambda n: f"{n}  ({model_paths[n]})",
            key="_select_base_model",
        )

        model_path = model_paths.get(base_model, "")
        model_classes = _read_model_classes(model_path)
        if model_classes:
            st.caption(f"This model detects **{len(model_classes)}** classes.")
        else:
            st.caption("Could not read classes from model metadata.")

        changed = base_model != saved_model
        btn_label = ":material/check: Confirm model" if changed else ":material/check: Model selected"
        if st.button(btn_label, type="primary" if changed else "secondary"):
            pdir = Path(st.session_state["workspace_dir"])
            meta_path = pdir / "project.json"
            meta = json.loads(meta_path.read_text())
            meta["base_model"] = base_model
            meta_path.write_text(json.dumps(meta, indent=2))

            st.session_state["base_model"] = base_model
            st.session_state["model_class_names"] = model_classes
            ds.set_setting("model_class_names", model_classes)
            st.rerun()


def _phase_select(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    _section_header("&#x1F4F7;", "Select Images")

    # --- Step 0: Base model ---
    _model_step(ds, args)

    images = ds.staged_images()
    has_images = len(images) > 0
    all_reviewed = (
        has_images
        and store.pending_count() == 0
        and store.reviewed_images_count() >= len(images)
    )

    # Show banner when classes need more training images
    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
    if needs:
        summary = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in needs
        )
        st.info(f"Classes needing more images: {summary}")

    # --- Step 1: Images ---
    images_detail = f"{len(images)} images" if has_images else ""
    with _step_expander("Images", done=has_images, detail=images_detail):
        if has_images:
            _images_management(ds, store, images)
            st.divider()
            st.subheader("Add more images")

        source = _data_source_radio(ds)
        _data_source_panels(source, ds, store, args)

    # --- Step 2: Review images ---
    with _step_expander("Review images", done=all_reviewed):
        if not has_images:
            st.info("Import images first.")
        elif all_reviewed:
            st.success("All images reviewed.")
        else:
            reviewed = store.reviewed_images_count()
            st.caption(f"Reviewed: {reviewed} / {len(images)}")
            if st.button(":material/rate_review: Go to Review", type="primary"):
                st.session_state["active_phase"] = "review"
                st.session_state.pop("_auto_label", None)
                st.rerun()
