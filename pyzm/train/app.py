"""Streamlit UI for YOLO fine-tuning (problem-driven workflow).

Launch with::

    python -m pyzm.train
    streamlit run pyzm/train/app.py -- --base-path /path/to/models

Phases (sidebar-driven):
    1. Select Images -- import frames from ZM events, YOLO datasets, or raw images
    2. Review Detections -- approve/edit/delete auto-detected objects
    3. Train & Export -- fine-tune and export ONNX
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import tempfile
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# streamlit-drawable-canvas 0.9.x calls st.elements.image.image_to_url which
# was removed in Streamlit >=1.39.  Provide a shim.
import streamlit.elements.image as _st_image
if not hasattr(_st_image, "image_to_url"):
    def _image_to_url(
        image, width, clamp, channels, output_format, image_id, **_kw
    ):
        from streamlit.runtime import get_instance
        buf = BytesIO()
        fmt = output_format or "PNG"
        image.save(buf, format=fmt)
        data = buf.getvalue()
        mimetype = f"image/{fmt.lower()}"
        mgr = get_instance().media_file_mgr
        return mgr.add(data, mimetype, image_id)
    _st_image.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas

from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.trainer import ClassMetrics, HardwareInfo, TrainProgress, TrainResult, YOLOTrainer
from st_clickable_images import clickable_images

from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")

DEFAULT_BASE_PATH = "/var/lib/zmeventnotification/models"
PROJECTS_ROOT = Path.home() / ".pyzm" / "training"
MIN_IMAGES_PER_CLASS = 10
_LOGO_PATH = Path(__file__).resolve().parents[2] / "logo" / "pyzm.png"

_COLOR_PALETTE = [
    "#27AE60", "#8E44AD", "#0081FE", "#FE3C71",
    "#F38630", "#5BB12F", "#E74C3C", "#3498DB",
]
_STATUS_COLORS = {
    DetectionStatus.PENDING: "#F1C40F",
    DetectionStatus.APPROVED: "#27AE60",
    DetectionStatus.DELETED: "#95A5A6",
    DetectionStatus.RENAMED: "#3498DB",
    DetectionStatus.RESHAPED: "#E67E22",
    DetectionStatus.ADDED: "#9B59B6",
}

_THUMB_WIDTH = 280
_THUMB_CACHE_MAX = 100


# ===================================================================
# Compact CSS
# ===================================================================

def _inject_css() -> None:
    st.markdown("""<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    section[data-testid="stSidebar"] { min-width: 260px; max-width: 300px; }
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left; width: 100%; font-size: 0.85em; padding: 0.3rem 0.5rem;
    }
    .stCaption { font-size: 0.78em !important; }
    .stCheckbox label, .stRadio label { font-size: 0.85em; }
    /* Hide Streamlit chrome but keep sidebar toggle */
    #MainMenu { display: none; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        height: auto !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] > * { pointer-events: auto; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { display: none; }
    /* Hide image/element toolbar (download, fullscreen, share, etc.) */
    [data-testid="stElementToolbar"] {
        display: none !important;
    }
    /* Section headers */
    .pyzm-section-header {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.5rem 0.75rem; margin: 0.75rem 0 0.5rem 0;
        border-left: 4px solid #4A90D9; background: rgba(74,144,217,0.08);
        border-radius: 0 0.4rem 0.4rem 0;
    }
    .pyzm-section-header h3 {
        margin: 0 !important; font-size: 1.1rem !important; font-weight: 600;
    }
    /* Project cards */
    .pyzm-project-card {
        padding: 0.2rem 0; margin: 0;
    }
    .pyzm-project-card .proj-name {
        font-size: 0.95rem; font-weight: 600; margin: 0;
    }
    .pyzm-project-card .meta { color: #94A3B8; font-size: 0.8rem; }
    </style>""", unsafe_allow_html=True)


def _section_header(icon: str, title: str) -> None:
    """Render a styled section header with a material icon."""
    st.markdown(
        f'<div class="pyzm-section-header">'
        f'<span style="font-size:1.3rem;">{icon}</span>'
        f'<h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


# ===================================================================
# Helpers
# ===================================================================

import re as _re

def _friendly_image_name(stem: str) -> str:
    """Turn filenames like ``event491_frame10`` into ``Event 491, Frame 10``."""
    m = _re.match(r"event(\d+)_frame(\d+)", stem)
    if m:
        return f"Event {m.group(1)}, Frame {m.group(2)}"
    return stem[:25]


# ===================================================================
# CLI args
# ===================================================================

def _parse_app_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    ap.add_argument("--workspace-dir", default=None)
    ap.add_argument("--processor", default="gpu")
    args, _ = ap.parse_known_args()
    return args


# ===================================================================
# Model scanning utilities
# ===================================================================

def _scan_models(base_path: str) -> list[dict]:
    bp = Path(base_path)
    if not bp.exists():
        return [{"name": "yolo11s", "path": "yolo11s.pt (auto-download)"}]
    models: list[dict] = []
    for f in sorted(bp.rglob("*.onnx")):
        if f.is_file():
            models.append({"name": f.stem, "path": str(f)})
    if not models:
        models.append({"name": "yolo11s", "path": "yolo11s.pt (auto-download)"})
    return models


def _read_model_classes(model_path: str) -> list[str]:
    p = Path(model_path)
    if not p.exists() or p.suffix != ".onnx":
        return []
    try:
        import onnx
        model = onnx.load(str(p))
        meta = {prop.key: prop.value for prop in model.metadata_props}
        if "names" in meta:
            try:
                names_dict = json.loads(meta["names"])
            except (json.JSONDecodeError, ValueError):
                import ast
                names_dict = ast.literal_eval(meta["names"])
            return [names_dict[i] for i in sorted(names_dict)]
    except Exception:
        pass
    return []



def _load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ===================================================================
# Drawing helpers
# ===================================================================

def _draw_verified_image(
    pil_img: Image.Image,
    detections: list[VerifiedDetection],
    *,
    skip_det_id: str | None = None,
) -> Image.Image:
    """Draw numbered, color-coded boxes on a PIL image.

    Parameters
    ----------
    skip_det_id
        If set, omit this detection (used during reshape mode so the
        editable rect isn't drawn twice).
    """
    from PIL import ImageDraw, ImageFont

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    num = 0
    for det in detections:
        if det.status == DetectionStatus.DELETED:
            continue
        num += 1
        if det.detection_id == skip_det_id:
            continue
        ann = det.effective_annotation
        color = _STATUS_COLORS.get(det.status, "#999999")
        label_text = f"#{num} {det.effective_label}"

        x1 = int((ann.cx - ann.w / 2) * w)
        y1 = int((ann.cy - ann.h / 2) * h)
        x2 = int((ann.cx + ann.w / 2) * w)
        y2 = int((ann.cy + ann.h / 2) * h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        bbox = font.getbbox(label_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 4)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=color)
        draw.text((x1 + 3, label_y + 1), label_text, fill="#FFFFFF", font=font)

    return img


# ===================================================================
# Grid review helpers
# ===================================================================

def _is_reviewed(store: VerificationStore, image_name: str) -> bool:
    """Check if an image is fully reviewed."""
    iv = store.get(image_name)
    return iv is not None and iv.fully_reviewed


def _filtered_images(
    images: list[Path],
    store: VerificationStore,
    filter_mode: str,
    object_class: str | None = None,
) -> list[Path]:
    """Filter images by review status and optionally by object class."""
    if filter_mode == "approved":
        result = [p for p in images if _is_reviewed(store, p.name)]
    elif filter_mode == "unapproved":
        result = [p for p in images if not _is_reviewed(store, p.name)]
    else:
        result = list(images)

    if object_class:
        result = [
            p for p in result
            if _image_has_class(store, p.name, object_class)
        ]
    return result


def _image_has_class(
    store: VerificationStore, image_name: str, class_name: str,
) -> bool:
    """Check if an image has at least one non-deleted detection of *class_name*."""
    iv = store.get(image_name)
    if iv is None:
        return False
    return any(
        d.effective_label == class_name
        for d in iv.detections
        if d.status != DetectionStatus.DELETED
    )


def _generate_thumbnail_uri(img_path: Path, store: VerificationStore) -> str:
    """Generate a thumbnail data URI with detection boxes and status border."""
    from PIL import ImageDraw, ImageFont

    cache: dict[str, str] = st.session_state.setdefault("_thumb_cache", {})
    name = img_path.name
    if name in cache:
        return cache[name]

    pil_img = _load_image_pil(img_path)
    orig_w, orig_h = pil_img.size
    scale = _THUMB_WIDTH / orig_w
    thumb_h = int(orig_h * scale)
    thumb = pil_img.resize((_THUMB_WIDTH, thumb_h))

    iv = store.get(name)
    if iv and iv.detections:
        thumb = _draw_verified_image(thumb, iv.detections)

    # Status border
    if iv and iv.fully_reviewed:
        border_color = "#27AE60"
    elif iv and iv.detections:
        border_color = "#F1C40F"
    else:
        border_color = None

    if border_color:
        draw = ImageDraw.Draw(thumb)
        w, h = thumb.size
        for i in range(3):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=border_color)

    # Caption bar
    det_count = len([
        d for d in (iv.detections if iv else [])
        if d.status != DetectionStatus.DELETED
    ])
    caption = _friendly_image_name(img_path.stem)
    if det_count:
        caption += f" ({det_count})"

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11,
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = font.getbbox(caption)
    text_h = bbox[3] - bbox[1]
    bar_h = text_h + 6
    w, h = thumb.size

    bar_img = Image.new("RGBA", (w, bar_h), (0, 0, 0, 160))
    thumb_rgba = thumb.convert("RGBA")
    thumb_rgba.paste(bar_img, (0, h - bar_h), bar_img)
    draw = ImageDraw.Draw(thumb_rgba)
    text_w = bbox[2] - bbox[0]
    tx = (w - text_w) // 2
    ty = h - bar_h + 2
    draw.text((tx, ty), caption, fill="#FFFFFF", font=font)

    buf = BytesIO()
    thumb_rgba.convert("RGB").save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    uri = f"data:image/jpeg;base64,{b64}"

    # FIFO cache eviction
    if len(cache) >= _THUMB_CACHE_MAX:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[name] = uri
    return uri


def _invalidate_thumbnail(image_name: str) -> None:
    """Remove a thumbnail from the cache so it's regenerated."""
    cache: dict[str, str] = st.session_state.get("_thumb_cache", {})
    cache.pop(image_name, None)


def _pagination_controls(page: int, total_pages: int, position: str) -> None:
    """Render Prev / Page X of Y / Next controls."""
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("Prev", disabled=page <= 0, key=f"page_prev_{position}"):
            st.session_state["_review_page"] = page - 1
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center; font-size:0.9em; padding:0.3rem 0;'>"
            f"Page {page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Next", disabled=page >= total_pages - 1,
                      key=f"page_next_{position}"):
            st.session_state["_review_page"] = page + 1
            st.rerun()


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
# SIDEBAR
# ===================================================================

def _sidebar(ds: YOLODataset | None, store: VerificationStore | None) -> str:
    """Render sidebar. Returns the active phase key."""
    with st.sidebar:
        st.image(str(_LOGO_PATH), width=120)

        # --- Phase navigation ---
        images = ds.staged_images() if ds else []
        has_images = len(images) >= 1
        all_reviewed = (
            has_images
            and store is not None
            and store.pending_count() == 0
            and store.reviewed_images_count() >= len(images)
        )
        train_done = st.session_state.get("_train_shared", {}).get("result") is not None

        current = st.session_state.get("active_phase", "select")

        phases = [
            ("select",  ":material/add_photo_alternate: Select Images",  has_images),
            ("review",  ":material/rate_review: Review Detections",      all_reviewed),
            ("train",   ":material/model_training: Train & Export",      train_done),
        ]
        for key, label, done in phases:
            prefix = "▸ " if key == current else "  "
            check = " ✓" if done else ""
            btn_label = f"{prefix}{label}{check}"
            if st.button(btn_label, key=f"phase_{key}", width="stretch"):
                st.session_state["active_phase"] = key
                st.session_state.pop("_auto_label", None)
                st.rerun()

        # --- Review progress (during review phase) ---
        if current == "review" and images and store:
            st.divider()
            st.markdown(":material/checklist: **Review Progress**")
            _sidebar_review_summary(store, len(images))

            # Bulk approve by confidence
            st.divider()
            st.markdown(":material/done_all: **Bulk Approve**")
            bulk_thresh = st.slider(
                "Min confidence", 0.5, 1.0, 0.7, 0.05,
                key="_bulk_approve_thresh",
                help="Approve all detections above this confidence.",
            )
            if st.button("Apply", key="bulk_approve_apply"):
                count = 0
                for img_name in store.all_images():
                    iv = store.get(img_name)
                    if iv is None:
                        continue
                    for det in iv.detections:
                        if det.status == DetectionStatus.PENDING:
                            conf = det.confidence
                            if conf is not None and conf >= bulk_thresh:
                                det.status = DetectionStatus.APPROVED
                                count += 1
                    if iv.pending_count == 0 and iv.detections:
                        iv.fully_reviewed = True
                    store.set(iv)
                store.save()
                st.session_state["_review_grid_counter"] = (
                    st.session_state.get("_review_grid_counter", 0) + 1
                )
                st.session_state["_thumb_cache"] = {}
                st.toast(f"Approved {count} detection(s)")
                st.rerun()

        # --- Class coverage ---
        if store:
            classes = store.build_class_list()
            if classes:
                st.divider()
                st.markdown(":material/category: **Class Coverage**")
                counts = store.per_class_image_counts(classes)
                for cls in classes:
                    count = counts.get(cls, 0)
                    pct = min(1.0, count / MIN_IMAGES_PER_CLASS)
                    ready = " ok" if count >= MIN_IMAGES_PER_CLASS else ""
                    st.caption(f"{cls}: {count}/{MIN_IMAGES_PER_CLASS}{ready}")
                    st.progress(pct)

        # --- Project actions ---
        st.divider()
        project_name = st.session_state.get("project_name", "")
        if project_name:
            st.markdown(f":material/folder_open: **{project_name}**")
            st.caption(f"`{PROJECTS_ROOT / project_name}`")
        if st.button(":material/swap_horiz: Switch Project", key="switch_project"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        if st.button(":material/restart_alt: Reset Project", key="reset_workspace"):
            st.session_state["_confirm_reset"] = True
            st.rerun()
        if st.session_state.get("_confirm_reset"):
            st.warning("Delete all images, annotations, and settings?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, reset", type="primary", key="reset_confirm"):
                    _reset_project()
            with c2:
                if st.button("Cancel", key="reset_cancel"):
                    st.session_state.pop("_confirm_reset", None)
                    st.rerun()

    return current


def _sidebar_review_summary(store: VerificationStore, total: int) -> None:
    """Review progress summary in the sidebar."""
    reviewed = store.reviewed_images_count()
    unapproved = total - reviewed
    st.caption(f"Approved: {reviewed} / {total}")
    if unapproved > 0:
        st.caption(f"Unapproved: {unapproved}")
    pct = reviewed / total if total > 0 else 0.0
    st.progress(pct)



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


# ===================================================================
# PHASE 2: Review Detections
# ===================================================================

def _phase_review(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    images = ds.staged_images()
    if not images:
        st.info("No images yet. Go to **Select Images** to add some.")
        return

    _section_header("&#x1F50D;", "Review Detections")

    with st.expander("Detection status colors"):
        legend_cols = st.columns(6)
        for col, (status, color) in zip(legend_cols, _STATUS_COLORS.items()):
            label_map = {
                DetectionStatus.PENDING: "Pending review",
                DetectionStatus.APPROVED: "Approved",
                DetectionStatus.DELETED: "Deleted",
                DetectionStatus.RENAMED: "Renamed",
                DetectionStatus.RESHAPED: "Reshaped",
                DetectionStatus.ADDED: "Added manually",
            }
            col.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{color};border-radius:2px;vertical-align:middle;'></span> "
                f"<span style='font-size:0.85em;'>{label_map.get(status, status.value)}</span>",
                unsafe_allow_html=True,
            )

    # Persistent auto-detect toggle (saved in project.json)
    auto_detect_default = bool(ds.get_setting("auto_detect", True))
    auto_detect = st.checkbox(
        "Automatically detect objects",
        value=auto_detect_default,
        key="_auto_detect_toggle",
        help="Run YOLO auto-detection when viewing an image with no annotations.",
    )
    if auto_detect != auto_detect_default:
        ds.set_setting("auto_detect", auto_detect)

    # --- Filter bar ---
    reviewed_count = store.reviewed_images_count()
    total = len(images)
    unapproved_count = total - reviewed_count

    all_classes = store.build_class_list()

    filter_col, obj_col, size_col = st.columns([3, 1, 1])
    with filter_col:
        filter_mode = st.radio(
            "Filter",
            ["all", "approved", "unapproved"],
            format_func=lambda x: {
                "all": f"All ({total})",
                "approved": f"Approved ({reviewed_count})",
                "unapproved": f"Unapproved ({unapproved_count})",
            }[x],
            horizontal=True,
            key="_review_filter",
        )
    with obj_col:
        object_class = st.selectbox(
            "Object",
            [""] + all_classes,
            format_func=lambda x: x or "All objects",
            key="_review_object_class",
        ) if all_classes else ""
    with size_col:
        page_size = st.selectbox(
            "Per page", [12, 20, 40, 60], index=1, key="_review_page_size",
        )

    # --- All-reviewed banner ---
    all_reviewed = reviewed_count >= total and store.pending_count() == 0
    if all_reviewed and "_review_expanded_idx" not in st.session_state:
        needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
        if needs:
            names = ", ".join(f"**{e['class_name']}**" for e in needs)
            st.info(
                f"All images reviewed. Classes needing more images: {names}."
            )
            if st.button("Import More Images", type="primary",
                         key="grid_go_select"):
                st.session_state["active_phase"] = "select"
                st.rerun()
        else:
            st.success("All images reviewed!")
            if st.button(":material/model_training: Continue to Train & Export", type="primary",
                         key="grid_go_train"):
                st.session_state["active_phase"] = "train"
                st.rerun()

    filtered = _filtered_images(images, store, filter_mode, object_class or None)
    if not filtered:
        st.info("No images match this filter.")
        return

    # Dispatch: expanded single-image view or thumbnail grid
    if "_review_expanded_idx" in st.session_state:
        _review_expanded(ds, store, args, filtered, auto_detect)
    else:
        _review_grid(store, filtered, page_size)


def _review_grid(
    store: VerificationStore, filtered: list[Path], page_size: int,
) -> None:
    """Paginated thumbnail grid of images."""
    total_pages = max(1, -(-len(filtered) // page_size))
    page = st.session_state.get("_review_page", 0)
    if page >= total_pages:
        page = total_pages - 1
        st.session_state["_review_page"] = page

    _pagination_controls(page, total_pages, "top")

    start = page * page_size
    end = min(start + page_size, len(filtered))
    page_images = filtered[start:end]

    paths: list[str] = []
    titles: list[str] = []
    for img_path in page_images:
        paths.append(_generate_thumbnail_uri(img_path, store))
        titles.append(img_path.stem)

    counter = st.session_state.get("_review_grid_counter", 0)
    clicked_idx = clickable_images(
        paths, titles=titles,
        div_style={
            "display": "flex", "flex-wrap": "wrap",
            "justify-content": "flex-start",
        },
        img_style={
            "width": "23%", "margin": "1%",
            "border-radius": "6px", "cursor": "pointer",
        },
        key=f"review_grid_{page}_{counter}",
    )

    if clicked_idx > -1 and clicked_idx < len(page_images):
        st.session_state["_review_expanded_idx"] = start + clicked_idx
        st.rerun()

    if total_pages > 1:
        _pagination_controls(page, total_pages, "bottom")


def _review_expanded(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
    filtered: list[Path],
    auto_detect: bool,
) -> None:
    """Single-image expanded view launched from the grid."""
    expanded_idx = st.session_state.get("_review_expanded_idx", 0)
    if expanded_idx >= len(filtered):
        st.session_state.pop("_review_expanded_idx", None)
        st.rerun()
        return

    img_path = filtered[expanded_idx]
    pil_img = _load_image_pil(img_path)
    iv = store.get(img_path.name)
    if iv is None:
        iv = ImageVerification(image_name=img_path.name)
        store.set(iv)

    # On-demand auto-detection
    _ran_autodetect = False
    if auto_detect and not iv.detections and not iv.fully_reviewed:
        _ran_autodetect = True
        detections = _auto_detect_image(img_path, args)
        if detections:
            iv.detections = detections
            store.set(iv)
            store.save()
            _invalidate_thumbnail(img_path.name)

    if _ran_autodetect and not iv.detections:
        st.info(
            "No objects detected automatically. This could mean the image is empty "
            "or the model doesn't recognize objects in this scene. "
            "Draw boxes manually to mark what you want the model to learn."
        )

    image_name = img_path.name
    canvas_counter = st.session_state.get(f"_canvas_counter_{image_name}", 0)

    # --- Navigation bar ---
    back_col, nav1, nav2, nav3 = st.columns([1, 1, 4, 1])
    with back_col:
        if st.button("< Grid", key="back_to_grid"):
            st.session_state.pop("_review_expanded_idx", None)
            st.rerun()
    with nav1:
        if st.button("Prev", disabled=expanded_idx == 0, key="expanded_prev"):
            st.session_state["_review_expanded_idx"] = expanded_idx - 1
            st.rerun()
    with nav2:
        status_text = "Reviewed" if iv.fully_reviewed else f"{iv.pending_count} pending"
        status_color = (
            _STATUS_COLORS[DetectionStatus.APPROVED]
            if iv.fully_reviewed
            else _STATUS_COLORS[DetectionStatus.PENDING]
        )
        st.markdown(
            f"<div style='text-align:center; font-size:0.9em;'>"
            f"<b>{img_path.name}</b> ({expanded_idx + 1}/{len(filtered)}) "
            f"<span style='color:{status_color}'>{status_text}</span></div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("Next", disabled=expanded_idx >= len(filtered) - 1,
                      key="expanded_next"):
            st.session_state["_review_expanded_idx"] = expanded_idx + 1
            st.rerun()

    # --- Compute canvas dimensions ---
    img_w, img_h = pil_img.size
    expanded = st.session_state.get("_canvas_expanded", False)
    max_w = 1200 if expanded else 700
    scale = min(1.0, max_w / img_w)
    canvas_w = int(img_w * scale)
    canvas_h = int(img_h * scale)

    # --- Check modal states ---
    reshape_det_id = st.session_state.get(f"_reshape_{image_name}")
    pending_rects: list[dict] = st.session_state.get(
        f"_pending_rects_{image_name}", [],
    )
    changed = False

    if reshape_det_id:
        # ---- RESHAPE MODE ----
        changed |= _canvas_reshape(
            pil_img, iv, image_name, reshape_det_id,
            canvas_w, canvas_h, canvas_counter,
        )
    elif pending_rects:
        # ---- LABEL NEW BOX MODE ----
        changed |= _canvas_label_pending(
            pil_img, iv, store, image_name, pending_rects,
            canvas_w, canvas_h, canvas_counter,
        )
    else:
        # ---- NORMAL MODE: interactive canvas ----
        tb1, tb2 = st.columns(2)
        with tb1:
            expand_label = "Shrink canvas" if expanded else "Expand canvas"
            if st.button(expand_label, key="toggle_canvas_expand"):
                st.session_state["_canvas_expanded"] = not expanded
                st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
                st.rerun()
        with tb2:
            if st.button("Clear drawn box", key="undo_canvas_draw"):
                st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
                st.rerun()

        st.info("Click and drag on the image to draw a box around objects the model missed.")

        auto_label = st.session_state.get("_auto_label")
        if auto_label:
            st.caption(f"Drawing as: **{auto_label}**")

        bg_img = pil_img.resize((canvas_w, canvas_h))
        bg_img = _draw_verified_image(bg_img, iv.detections)

        canvas_result = st_canvas(
            fill_color="#9B59B622",
            stroke_width=3,
            stroke_color="#9B59B6",
            background_image=bg_img,
            drawing_mode="rect",
            height=canvas_h,
            width=canvas_w,
            key=f"canvas_{image_name}_{canvas_counter}",
        )

        # Detect newly drawn rectangles
        if canvas_result and canvas_result.json_data:
            new_rects = [
                obj for obj in canvas_result.json_data.get("objects", [])
                if obj.get("type") == "rect"
            ]
            if new_rects:
                if auto_label:
                    _save_pending_rects(
                        iv, new_rects, auto_label,
                        canvas_w, canvas_h, image_name, canvas_counter,
                    )
                    changed = True
                else:
                    st.session_state[f"_pending_rects_{image_name}"] = new_rects
                    st.rerun()

    # --- Detection list (always visible unless reshaping) ---
    if not reshape_det_id:
        if iv.detections and not pending_rects:
            changed |= _detection_list(iv, store, image_name)

    # --- Primary action button ---
    if not reshape_det_id and not pending_rects:
        st.divider()
        pending = [d for d in iv.detections if d.status == DetectionStatus.PENDING]
        is_last = expanded_idx >= len(filtered) - 1
        filter_mode = st.session_state.get("_review_filter", "all")

        if pending:
            btn_label = (
                f"Approve all ({len(pending)}) & back to grid"
                if is_last
                else f"Approve all ({len(pending)}) & next"
            )
        else:
            btn_label = "Next image" if not is_last else "Back to grid"

        if st.button(btn_label, type="primary", width="stretch"):
            for d in iv.detections:
                if d.status == DetectionStatus.PENDING:
                    d.status = DetectionStatus.APPROVED
            iv.fully_reviewed = True
            store.set(iv)
            store.save()
            _invalidate_thumbnail(image_name)
            st.session_state["_review_grid_counter"] = (
                st.session_state.get("_review_grid_counter", 0) + 1
            )

            if is_last:
                st.session_state.pop("_review_expanded_idx", None)
            elif filter_mode == "unapproved":
                pass  # same index, list shifts
            else:
                st.session_state["_review_expanded_idx"] = expanded_idx + 1
            st.rerun()

    # --- Re-review & Remove image ---
    if not reshape_det_id and not pending_rects:
        re_col, rm_col = st.columns(2)
        with re_col:
            if iv.fully_reviewed:
                if st.button(":material/undo: Re-review", key=f"rereview_{image_name}"):
                    for d in iv.detections:
                        if d.status != DetectionStatus.DELETED:
                            d.status = DetectionStatus.PENDING
                    iv.fully_reviewed = False
                    changed = True
        with rm_col:
            if st.button(":material/delete: Remove Image", key=f"rmimg_{image_name}"):
                st.session_state[f"_confirm_rmimg_{image_name}"] = True
                st.rerun()
        if st.session_state.get(f"_confirm_rmimg_{image_name}"):
            st.warning(f"Remove **{image_name}** and its annotations?")
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("Yes, remove", type="primary", key=f"rmimg_yes_{image_name}"):
                    # Remove image and label files
                    img_file = ds._images_dir / image_name
                    label_file = ds._labels_dir / (img_path.stem + ".txt")
                    if img_file.exists():
                        img_file.unlink()
                    if label_file.exists():
                        label_file.unlink()
                    store.remove(image_name)
                    store.save()
                    _invalidate_thumbnail(image_name)
                    st.session_state.pop(f"_confirm_rmimg_{image_name}", None)
                    st.session_state.pop("_review_expanded_idx", None)
                    st.session_state["_review_grid_counter"] = (
                        st.session_state.get("_review_grid_counter", 0) + 1
                    )
                    st.session_state["_thumb_cache"] = {}
                    st.toast(f"Removed {image_name}")
                    st.rerun()
            with no_col:
                if st.button("Cancel", key=f"rmimg_no_{image_name}"):
                    st.session_state.pop(f"_confirm_rmimg_{image_name}", None)
                    st.rerun()

    if changed:
        store.set(iv)
        store.save()
        _invalidate_thumbnail(image_name)
        st.session_state["_review_grid_counter"] = (
            st.session_state.get("_review_grid_counter", 0) + 1
        )
        st.rerun()

    # --- Post-review guidance ---
    all_images = ds.staged_images()
    all_reviewed = (
        store.pending_count() == 0
        and store.reviewed_images_count() >= len(all_images)
    )
    if all_reviewed and not reshape_det_id and not pending_rects:
        needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
        st.divider()
        if needs:
            names = ", ".join(f"**{e['class_name']}**" for e in needs)
            st.info(
                f"Classes needing more training images: {names}. "
                f"Upload at least **{MIN_IMAGES_PER_CLASS}** images for each."
            )
            if st.button("Import More Images", type="primary",
                         key="review_go_select"):
                st.session_state["active_phase"] = "select"
                st.session_state.pop("_review_expanded_idx", None)
                st.rerun()
        else:
            st.success("All images reviewed! Ready to train.")
            if st.button(":material/model_training: Continue to Train", type="primary",
                         key="review_go_train"):
                st.session_state["active_phase"] = "train"
                st.session_state.pop("_review_expanded_idx", None)
                st.rerun()


# -------------------------------------------------------------------
# Canvas sub-modes
# -------------------------------------------------------------------

def _canvas_reshape(
    pil_img: Image.Image,
    iv: ImageVerification,
    image_name: str,
    det_id: str,
    canvas_w: int,
    canvas_h: int,
    canvas_counter: int,
) -> bool:
    """Reshape mode: canvas in ``transform`` with the selected detection
    as a draggable/resizable rectangle."""
    det = next((d for d in iv.detections if d.detection_id == det_id), None)
    if det is None:
        st.session_state.pop(f"_reshape_{image_name}", None)
        return False

    # Background: all detections EXCEPT the one being reshaped
    bg_img = pil_img.resize((canvas_w, canvas_h))
    bg_img = _draw_verified_image(bg_img, iv.detections, skip_det_id=det_id)

    # Initial drawing: the detection to reshape
    ann = det.effective_annotation
    x = int((ann.cx - ann.w / 2) * canvas_w)
    y = int((ann.cy - ann.h / 2) * canvas_h)
    w = int(ann.w * canvas_w)
    h = int(ann.h * canvas_h)

    initial = {
        "version": "4.4.0",
        "objects": [{
            "type": "rect",
            "left": x, "top": y, "width": w, "height": h,
            "fill": "rgba(155,89,182,0.12)",
            "stroke": "#E67E22", "strokeWidth": 3,
            "scaleX": 1, "scaleY": 1,
        }],
    }

    st.markdown(
        f"<div style='font-size:0.85em; margin-bottom:4px;'>"
        f"Reshaping <b>#{_visible_index(iv, det_id)} {det.effective_label}</b> "
        f"&mdash; drag or resize the box, then save.</div>",
        unsafe_allow_html=True,
    )

    canvas_result = st_canvas(
        fill_color="#9B59B622",
        stroke_width=3,
        stroke_color="#E67E22",
        background_image=bg_img,
        drawing_mode="transform",
        initial_drawing=initial,
        height=canvas_h,
        width=canvas_w,
        key=f"canvas_reshape_{image_name}_{canvas_counter}",
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save shape", type="primary", width="stretch"):
            # Read the updated rect from canvas
            objs = (canvas_result.json_data or {}).get("objects", [])
            rect = next((o for o in objs if o.get("type") == "rect"), None)
            if rect:
                left = rect["left"]
                top = rect["top"]
                rw = rect["width"] * rect.get("scaleX", 1)
                rh = rect["height"] * rect.get("scaleY", 1)
                cx = max(0.0, min(1.0, (left + rw / 2) / canvas_w))
                cy = max(0.0, min(1.0, (top + rh / 2) / canvas_h))
                nw = max(0.0, min(1.0, rw / canvas_w))
                nh = max(0.0, min(1.0, rh / canvas_h))
                det.adjusted = Annotation(class_id=0, cx=cx, cy=cy, w=nw, h=nh)
                det.status = DetectionStatus.RESHAPED
            st.session_state.pop(f"_reshape_{image_name}", None)
            st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
            return True
    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.session_state.pop(f"_reshape_{image_name}", None)
            st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
            st.rerun()

    return False


def _canvas_label_pending(
    pil_img: Image.Image,
    iv: ImageVerification,
    store: VerificationStore,
    image_name: str,
    pending_rects: list[dict],
    canvas_w: int,
    canvas_h: int,
    canvas_counter: int,
) -> bool:
    """Show the image with the newly drawn box and a label-picker dialog."""
    from PIL import ImageDraw as _IDraw

    bg_img = pil_img.resize((canvas_w, canvas_h))
    bg_img = _draw_verified_image(bg_img, iv.detections)
    preview = bg_img.copy()
    drw = _IDraw.Draw(preview)
    for rect in pending_rects:
        x1 = int(rect["left"])
        y1 = int(rect["top"])
        x2 = x1 + int(rect["width"] * rect.get("scaleX", 1))
        y2 = y1 + int(rect["height"] * rect.get("scaleY", 1))
        drw.rectangle([x1, y1, x2, y2], outline="#9B59B6", width=3)
    st.image(preview, width=canvas_w)

    # Label dialog — text input + selectbox side by side; text input wins
    changed = False

    model_classes = st.session_state.get("model_class_names", [])
    user_labels = store.build_class_list()
    all_labels = sorted(set(model_classes) | set(user_labels))

    with st.form(key=f"lbl_form_{image_name}_{canvas_counter}"):
        c_new, c_existing = st.columns(2)
        with c_new:
            typed_label = st.text_input("Type a label name", placeholder="e.g. dog, car...")
        with c_existing:
            picked_label = st.selectbox(
                "Select existing label",
                options=[""] + all_labels,
                format_func=lambda x: x or "—",
            ) if all_labels else ""
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Apply", type="primary", width="stretch")
        with c2:
            cancelled = st.form_submit_button("Cancel", width="stretch")

    if submitted:
        final = typed_label.strip() if typed_label and typed_label.strip() else (picked_label or "")
        if final:
            _save_pending_rects(iv, pending_rects, final,
                               canvas_w, canvas_h, image_name, canvas_counter)
            changed = True
    elif cancelled:
        st.session_state[f"_pending_rects_{image_name}"] = []
        st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
        st.rerun()

    return changed


def _save_pending_rects(
    iv: ImageVerification,
    rects: list[dict],
    label: str,
    canvas_w: int,
    canvas_h: int,
    image_name: str,
    canvas_counter: int,
) -> None:
    """Convert pending canvas rectangles into VerifiedDetections and clear state."""
    existing_count = len(iv.detections)
    for j, rect in enumerate(rects):
        left = rect["left"]
        top = rect["top"]
        w = rect["width"] * rect.get("scaleX", 1)
        h = rect["height"] * rect.get("scaleY", 1)
        cx = max(0.0, min(1.0, (left + w / 2) / canvas_w))
        cy = max(0.0, min(1.0, (top + h / 2) / canvas_h))
        nw = max(0.0, min(1.0, w / canvas_w))
        nh = max(0.0, min(1.0, h / canvas_h))
        ann = Annotation(class_id=0, cx=cx, cy=cy, w=nw, h=nh)
        iv.detections.append(VerifiedDetection(
            detection_id=f"det_{existing_count + j}",
            original=ann,
            status=DetectionStatus.ADDED,
            original_label=label,
        ))
    st.session_state[f"_pending_rects_{image_name}"] = []
    st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
    st.toast(f"Added {len(rects)} detection{'s' if len(rects) != 1 else ''}: {label}")


def _visible_index(iv: ImageVerification, det_id: str) -> int:
    """Return the 1-based display number for a detection (skipping DELETED)."""
    num = 0
    for d in iv.detections:
        if d.status == DetectionStatus.DELETED:
            continue
        num += 1
        if d.detection_id == det_id:
            return num
    return 0


def _detection_list(
    iv: ImageVerification,
    store: VerificationStore,
    image_name: str,
) -> bool:
    """Numbered detection list with actions.  Numbers match the image overlay."""
    changed = False
    known_labels = store.build_class_list()
    num = 0

    for det in iv.detections:
        if det.status == DetectionStatus.DELETED:
            # Show deleted ones dimmed, with undo
            st.markdown(
                f"<div style='font-size:0.8em; color:#95A5A6; margin:2px 0;'>"
                f"<s>{det.effective_label}</s> deleted</div>",
                unsafe_allow_html=True,
            )
            if st.button("Undo", key=f"undo_{image_name}_{det.detection_id}"):
                det.status = DetectionStatus.PENDING
                changed = True
            continue

        num += 1
        sc = _STATUS_COLORS.get(det.status, "#999")

        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; margin:6px 0 2px;'>"
            f"<span style='display:inline-block; width:12px; height:12px; "
            f"background:{sc}; border-radius:2px;'></span>"
            f"<span style='font-size:0.85em;'><b>#{num} {det.effective_label}</b> "
            f"<span style='color:{sc}'>[{det.status.value}]</span></span></div>",
            unsafe_allow_html=True,
        )

        col_a, col_d, col_s, col_r = st.columns(4)
        with col_a:
            if det.status in (DetectionStatus.APPROVED, DetectionStatus.ADDED):
                st.markdown(
                    "<span style='color:#27AE60; font-size:0.8em;'>Approved</span>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button("Approve", key=f"approve_{image_name}_{det.detection_id}",
                             width="stretch"):
                    det.status = DetectionStatus.APPROVED
                    changed = True
        with col_d:
            if st.button("Delete", key=f"delete_{image_name}_{det.detection_id}",
                         width="stretch"):
                det.status = DetectionStatus.DELETED
                changed = True
        with col_s:
            if st.button("Reshape", key=f"reshape_{image_name}_{det.detection_id}",
                         width="stretch"):
                st.session_state[f"_reshape_{image_name}"] = det.detection_id
                st.rerun()
        with col_r:
            rename_key = f"_renaming_{image_name}_{det.detection_id}"
            if st.button("Rename", key=f"rename_btn_{image_name}_{det.detection_id}",
                         width="stretch"):
                st.session_state[rename_key] = True
                st.rerun()

        # Rename input row (shown only when Rename is clicked)
        if st.session_state.get(f"_renaming_{image_name}_{det.detection_id}"):
            other_labels = sorted(set(known_labels) - {det.effective_label})

            with st.form(key=f"rename_form_{image_name}_{det.detection_id}"):
                c_new, c_existing = st.columns(2)
                with c_new:
                    ren_typed = st.text_input("Type a label name", placeholder="e.g. dog, car...")
                with c_existing:
                    ren_picked = st.selectbox(
                        "Select existing label",
                        options=[""] + other_labels,
                        format_func=lambda x: x or "—",
                    ) if other_labels else ""
                rc1, rc2 = st.columns(2)
                with rc1:
                    submitted = st.form_submit_button("Save", type="primary", width="stretch")
                with rc2:
                    cancel_rename = st.form_submit_button("Cancel", width="stretch")

            if submitted:
                final = ren_typed.strip() if ren_typed and ren_typed.strip() else (ren_picked or "")
                if final:
                    det.new_label = final
                    det.status = DetectionStatus.RENAMED
                    st.session_state.pop(f"_renaming_{image_name}_{det.detection_id}", None)
                    changed = True
            elif cancel_rename:
                st.session_state.pop(f"_renaming_{image_name}_{det.detection_id}", None)
                st.rerun()

    return changed


# ===================================================================
# PHASE 3: Train & Export
# ===================================================================

def _phase_train(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    _section_header("&#x1F3AF;", "Train & Export")
    pdir = st.session_state.get("workspace_dir")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    classes = store.build_class_list()
    if not classes:
        st.info("No verified detections yet. Go to **Review Detections** first.")
        return

    # Readiness check — only corrected classes need the image threshold
    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
    if needs:
        names = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in needs
        )
        st.warning(f"Need more images for: {names}")
        if st.button("Import More Images", key="train_go_select"):
            st.session_state["active_phase"] = "select"
            st.rerun()

    images = ds.staged_images()
    if len(images) < 2:
        st.info(
            f"You have {len(images)} image(s). We recommend at least "
            f"10\u201320 images per object type for effective training."
        )
        return

    hw = YOLOTrainer.detect_hardware()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        epochs = st.number_input(
            "Epochs", min_value=1, max_value=300, value=50, step=10,
            help="How many times to review all images during training. 50 is usually good for fine-tuning.",
        )
    with col2:
        batch = st.number_input(
            "Batch", min_value=1, max_value=128, value=hw.suggested_batch,
            help="Images processed at once. Auto-detected for your hardware.",
        )
    with col3:
        imgsz = st.selectbox(
            "Image size", [416, 640], index=1,
            help="Resolution for training. 640 is standard.",
        )
    with col4:
        st.caption(hw.display)

    # Shared mutable dict — background thread mutates contents,
    # main thread reads.  Lives in session_state so it survives reruns.
    shared: dict = st.session_state.get("_train_shared", {})

    if not shared.get("active", False):
        all_ready = not needs
        st.info(
            f"This fine-tuned model will **only** detect the following "
            f"{len(classes)} object(s): **{', '.join(classes)}**. "
            f"It will not retain the base model's original classes. "
            f"To detect other objects, run the base model alongside this one."
        )
        if hw.device == "cpu":
            st.warning(
                "Training on CPU will be significantly slower than GPU. "
                "Consider using a machine with a CUDA-capable GPU for faster training."
            )
        # Training duration estimate
        hw_factor = 1 if hw.device != "cpu" else 8
        est_minutes = int((len(images) * epochs) / (batch * 60) * hw_factor)
        if est_minutes > 0:
            st.caption(f":material/timer: Estimated training time: ~{est_minutes} minutes (rough estimate)")
        if st.button(":material/rocket_launch: Start Training", type="primary", disabled=not all_ready):
            class_name_to_id = {c: i for i, c in enumerate(classes)}
            ds.set_classes(classes)
            yaml_path = ds.generate_yaml()

            trainer = YOLOTrainer(
                base_model=base_model,
                project_dir=Path(pdir),
                device=hw.device,
            )
            shared = {
                "active": True,
                "progress": TrainProgress(
                    total_epochs=epochs, message="Preparing dataset...",
                ),
                "result": None,
                "log": [],
            }
            import time as _time
            shared["start_time"] = _time.time()
            shared["image_count"] = len(images)
            st.session_state["_train_shared"] = shared
            st.session_state["trainer"] = trainer
            st.session_state["classes"] = classes

            def _run(_s: dict = shared) -> None:
                # ── Dataset preparation (runs in background) ──
                # Skip rewriting annotations when the dataset was
                # imported with the same classes and nothing was modified.
                import_classes = ds.get_setting("import_classes")
                need_rewrite = (
                    store.has_modifications()
                    or import_classes != classes
                )

                n = len(images)
                if need_rewrite:
                    for i, img_path in enumerate(images):
                        anns = store.finalized_annotations(
                            img_path.name, class_name_to_id,
                        )
                        ds.update_annotations(img_path.name, anns)
                        if n > 100 and (i + 1) % 50 == 0:
                            _s["progress"] = TrainProgress(
                                total_epochs=epochs,
                                message=f"Writing annotations {i + 1}/{n}",
                            )

                # Re-split only if annotations changed or no split exists yet
                split_map = ds.get_setting("split_map") or {}
                has_split = ds._train_images.exists() and any(ds._train_images.iterdir())
                if need_rewrite or not has_split:
                    _s["progress"] = TrainProgress(
                        total_epochs=epochs,
                        message="Splitting into train/val...",
                    )
                    ds.split()

                _s["progress"] = TrainProgress(
                    total_epochs=epochs, message="Loading model...",
                )

                # ── Training ──
                def _cb(p: TrainProgress) -> None:
                    _s["progress"] = p

                # Capture all output (logging + stdout/stderr)
                class _TrainLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        log = _s["log"]
                        log.append(self.format(record))
                        if len(log) > 200:
                            del log[:-200]

                class _StreamCapture:
                    """Tee writes to the original stream and the shared log."""
                    def __init__(self, original):
                        self._original = original
                    def write(self, text):
                        self._original.write(text)
                        if text.strip():
                            log = _s["log"]
                            log.append(text.rstrip())
                            if len(log) > 200:
                                del log[:-200]
                    def flush(self):
                        self._original.flush()
                    def __getattr__(self, name):
                        return getattr(self._original, name)

                handler = _TrainLogHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                ul_logger = logging.getLogger("ultralytics")
                ul_logger.addHandler(handler)
                ul_logger.setLevel(logging.INFO)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = _StreamCapture(old_stdout)
                sys.stderr = _StreamCapture(old_stderr)
                try:
                    r = trainer.train(
                        dataset_yaml=yaml_path, epochs=epochs,
                        batch=batch, imgsz=imgsz, progress_callback=_cb,
                    )
                    _s["result"] = r
                except Exception as exc:
                    _s["progress"] = TrainProgress(finished=True, error=str(exc))
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    ul_logger.removeHandler(handler)
                    _s["active"] = False

            threading.Thread(target=_run, daemon=True).start()
            st.rerun()
    else:
        st.warning("Do not close this browser tab while training is in progress. Training will be lost.")
        p: TrainProgress = shared.get("progress") or TrainProgress()
        if p.total_epochs > 0:
            pct = max(0.0, min(1.0, p.epoch / p.total_epochs))
            st.progress(pct, text=p.message or f"Epoch {p.epoch}/{p.total_epochs}")

        import time as _time
        start = shared.get("start_time")
        if start:
            elapsed = _time.time() - start
            mins, secs = divmod(int(elapsed), 60)
            hrs, mins = divmod(mins, 60)
            elapsed_str = f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins}:{secs:02d}"
        else:
            elapsed_str = "0:00"
        img_count = shared.get("image_count", 0)

        t1, t2 = st.columns(2)
        t1.markdown(f":material/timer: **Elapsed Time:** {elapsed_str}")
        t2.markdown(f":material/image: **Training Images:** {img_count}")

        # User-friendly metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")

        # Box accuracy trend
        prev_box = st.session_state.get("_prev_box_loss")
        if prev_box is not None and p.box_loss > 0:
            if p.box_loss < prev_box - 0.001:
                box_label = "Box accuracy: Improving"
                box_color = "green"
            elif p.box_loss > prev_box + 0.001:
                box_label = "Box accuracy: Declining"
                box_color = "red"
            else:
                box_label = "Box accuracy: Stable"
                box_color = "orange"
        else:
            box_label = "Box accuracy: Starting"
            box_color = "gray"
        if p.epoch > 0:
            st.session_state["_prev_box_loss"] = p.box_loss
        m2.markdown(f":{box_color}[{box_label}]")

        # Detection quality from mAP50
        mAP_pct = p.mAP50 * 100
        if mAP_pct >= 80:
            quality = "Excellent"
        elif mAP_pct >= 60:
            quality = "Good"
        elif mAP_pct >= 30:
            quality = "Fair"
        else:
            quality = "Poor"
        m3.metric("Detection quality", f"{mAP_pct:.0f}% ({quality})")

        with st.expander("Technical Details"):
            td1, td2, td3, td4 = st.columns(4)
            td1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")
            td2.metric("Box Loss", f"{p.box_loss:.4f}")
            td3.metric("Cls Loss", f"{p.cls_loss:.4f}")
            td4.metric("mAP50", f"{p.mAP50:.3f}")

        if st.button(":material/stop_circle: Stop Training"):
            trainer = st.session_state.get("trainer")
            if trainer:
                trainer.request_stop()

        # Live training log
        train_log = shared.get("log", [])
        if train_log:
            st.code("\n".join(train_log[-30:]), language=None)

        if p.error:
            st.error(p.error)
        elif not p.finished:
            import time
            time.sleep(2)
            st.rerun()

    # Results + Export
    result: TrainResult | None = shared.get("result")
    if result:
        st.markdown(
            '<div style="background:#1B5E20;padding:0.8rem 1rem;'
            'border-radius:0.5rem;margin:1rem 0;">'
            '<h2 style="color:white;text-align:center;margin:0;">'
            '&#x2705; Training Complete</h2></div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("mAP50", f"{result.final_mAP50:.3f}")
        r2.metric("mAP50-95", f"{result.final_mAP50_95:.3f}")
        r3.metric("Size", f"{result.model_size_mb:.1f} MB")
        r4.metric("Time", f"{result.elapsed_seconds / 60:.1f} min")

        best_ep_label = (
            f"Epoch {result.best_epoch}/{result.total_epochs}"
            if result.best_epoch > 0
            else f"{result.total_epochs} epochs"
        )
        st.caption(f":material/star: Best model from: **{best_ep_label}**")

        if result.per_class:
            import pandas as pd

            rows = []
            for cls_name, cm in sorted(result.per_class.items()):
                rows.append({
                    "Class": cls_name,
                    "Precision": f"{cm.precision:.3f}",
                    "Recall": f"{cm.recall:.3f}",
                    "AP@50": f"{cm.ap50:.3f}",
                    "AP@50-95": f"{cm.ap50_95:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if result.best_model:
            st.code(str(result.best_model), language=None)

        pdir = st.session_state.get("workspace_dir")
        if pdir:
            st.caption(f"Dataset: `{pdir}`")
            _training_analysis(result, Path(pdir) / "runs" / "train")

        _phase_export(args)


def _training_analysis(result: TrainResult, train_dir: Path) -> None:
    """Show interpretive guidance and diagnostic images after training."""
    _section_header("&#x1F4CA;", "Training Analysis")

    # -- (a) Interpretive guidance --
    mAP = result.final_mAP50
    if mAP >= 0.8:
        st.success(f"**Excellent** — mAP50 {mAP:.2f} indicates strong detection quality.")
    elif mAP >= 0.6:
        st.info(f"**Good** — mAP50 {mAP:.2f}. Model performs well; more data may push it higher.")
    elif mAP >= 0.3:
        st.warning(f"**Moderate** — mAP50 {mAP:.2f}. Consider adding more diverse training images.")
    else:
        st.warning(f"**Poor** — mAP50 {mAP:.2f}. The model needs significantly more training data.")

    # Per-class weak spots
    weak = [name for name, cm in result.per_class.items() if cm.ap50 < 0.5]
    if weak:
        st.info(
            f"**Weak classes** (AP50 < 0.5): {', '.join(sorted(weak))}. "
            "Consider adding more training images for these."
        )

    # Overfitting hint
    if (
        result.best_epoch > 0
        and result.total_epochs > 0
        and result.best_epoch < result.total_epochs * 0.5
    ):
        st.info(
            f"Best model was at epoch {result.best_epoch}/{result.total_epochs} "
            "(early in training). The model may have overfit — "
            "try fewer epochs or more training data."
        )

    # -- (b) Training curves --
    results_png = train_dir / "results.png"
    if results_png.exists():
        st.markdown("##### Training Curves")
        with st.expander("How to read these curves"):
            st.markdown(
                "**Loss curves** (box_loss, cls_loss, dfl_loss): should decrease "
                "and flatten. If training loss keeps dropping but validation loss "
                "rises, the model is overfitting — stop earlier or add more data.\n\n"
                "**Precision & Recall**: both should rise and stabilize near 1.0. "
                "Low precision = too many false detections; low recall = missing "
                "real objects.\n\n"
                "**mAP50 / mAP50-95**: the main quality scores — higher is better. "
                "A plateau means more epochs won't help; more diverse data will."
            )
        st.image(str(results_png), width="stretch")

    # -- (c) Confusion matrix --
    cm_norm = train_dir / "confusion_matrix_normalized.png"
    cm_plain = train_dir / "confusion_matrix.png"
    cm_path = cm_norm if cm_norm.exists() else cm_plain if cm_plain.exists() else None
    if cm_path:
        st.markdown("##### Which Objects Get Mixed Up")
        with st.expander("How to read the confusion matrix"):
            st.markdown(
                "Each row is a **true** class, each column is a **predicted** class. "
                "Bright diagonal = correct predictions. Off-diagonal cells show "
                "which classes get confused with each other.\n\n"
                "A **background** row/column means missed detections (false negatives) "
                "or phantom detections (false positives). If a class has a high "
                "background score, the model needs more examples of that class."
            )
        st.image(str(cm_path), width="stretch")

    # -- (d) Evaluation curves --
    f1_path = train_dir / "F1_curve.png"
    pr_path = train_dir / "PR_curve.png"
    if f1_path.exists() or pr_path.exists():
        with st.expander("Detection Balance"):
            st.markdown(
                "**F1 Curve**: shows the balance between precision and recall at "
                "each confidence threshold. The peak is the optimal threshold — "
                "a sharp, high peak (close to 1.0) is ideal.\n\n"
                "**PR Curve**: precision vs. recall trade-off. A curve that hugs "
                "the top-right corner is a strong model. Area under the curve "
                "(AUC) equals AP — higher is better."
            )
            c1, c2 = st.columns(2)
            if f1_path.exists():
                c1.image(str(f1_path), caption="F1 Curve", width="stretch")
            if pr_path.exists():
                c2.image(str(pr_path), caption="PR Curve", width="stretch")

    # -- (e) Validation samples --
    val_labels = train_dir / "val_batch0_labels.jpg"
    val_preds = train_dir / "val_batch0_pred.jpg"
    if val_labels.exists() or val_preds.exists():
        with st.expander("Validation Samples"):
            st.markdown(
                "**Ground Truth** shows the actual labels. **Predictions** shows "
                "what the model detected. Compare them — missed objects or wrong "
                "labels indicate classes that need more training data."
            )
            c1, c2 = st.columns(2)
            if val_labels.exists():
                c1.image(str(val_labels), caption="Ground Truth", width="stretch")
            if val_preds.exists():
                c2.image(str(val_preds), caption="Predictions", width="stretch")


def _phase_export(args: argparse.Namespace) -> None:
    _section_header("&#x1F4E6;", "Export Model")
    pdir = st.session_state.get("workspace_dir")
    classes = st.session_state.get("classes", [])
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    best_pt = Path(pdir) / "runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        return

    st.info(
        f"This fine-tuned model only detects: **{', '.join(classes)}**. "
        f"To also detect standard objects (person, car, etc.), add both models "
        f"to your `objectconfig.yml` and use `same_model_sequence_strategy: union` "
        f"to merge results:\n"
        f"```yaml\n"
        f"same_model_sequence_strategy: union\n"
        f"```\n"
        f"Then list your base model and this fine-tuned model as separate entries "
        f"under the `object` sequence."
    )

    suggested_name = f"{base_model}_finetune.onnx"
    export_path = st.text_input(
        "Export ONNX to",
        value=str(Path(args.base_path) / "custom_finetune" / suggested_name),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(":material/download: Export ONNX", type="primary"):
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            with st.spinner("Exporting..."):
                try:
                    onnx_path = trainer.export_onnx(output_path=Path(export_path))
                    st.success(f"Exported: `{onnx_path}`")
                    st.code(
                        f"models:\n"
                        f"  - name: {onnx_path.stem}\n"
                        f"    type: object\n"
                        f"    framework: opencv\n"
                        f"    weights: {onnx_path}\n"
                        f"    min_confidence: 0.3\n"
                        f"    pattern: \"({'|'.join(classes)})\"\n",
                        language="yaml",
                    )
                    st.caption(
                        "Add this to your `objectconfig.yml` (usually at `/etc/zm/`) "
                        "under the `models:` section."
                    )
                except Exception as exc:
                    st.error(str(exc))

    with col2:
        test_file = st.file_uploader("Test image", type=["jpg", "jpeg", "png"], key="test_img")
        if test_file:
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            pil_img = Image.open(test_file).convert("RGB")
            img_array = np.array(pil_img)
            try:
                dets = trainer.evaluate(img_array[..., ::-1])
                from PIL import ImageDraw
                draw_img = pil_img.copy()
                draw = ImageDraw.Draw(draw_img)
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                    draw.text((x1, max(0, y1 - 12)), f"{d['label']} {d['confidence']:.0%}", fill="lime")
                st.image(draw_img, width="stretch")
                if dets:
                    st.caption(", ".join(f"{d['label']} {d['confidence']:.0%}" for d in dets))
                else:
                    st.caption("No detections")
            except Exception as exc:
                st.error(str(exc))


# ===================================================================
# Legacy label seeding
# ===================================================================

def _seed_from_legacy_labels(ds: YOLODataset, store: VerificationStore) -> None:
    images = ds.staged_images()
    classes = ds.classes
    seeded = 0
    for img_path in images:
        if store.get(img_path.name) is not None:
            continue
        anns = ds.annotations_for(img_path.name)
        if not anns:
            store.set(ImageVerification(image_name=img_path.name))
            continue
        detections = []
        for j, ann in enumerate(anns):
            label = classes[ann.class_id] if ann.class_id < len(classes) else f"class_{ann.class_id}"
            detections.append(VerifiedDetection(
                detection_id=f"det_{j}",
                original=ann,
                status=DetectionStatus.APPROVED,
                original_label=label,
            ))
        store.set(ImageVerification(
            image_name=img_path.name,
            detections=detections,
            fully_reviewed=True,
        ))
        seeded += 1
    if seeded:
        store.save()
        logger.info("Seeded %d images from legacy labels into VerificationStore", seeded)


# ===================================================================
# Main
# ===================================================================

def _reset_project() -> None:
    """Wipe current project and return to project selector."""
    import shutil

    pdir = st.session_state.get("workspace_dir")
    if pdir and Path(pdir).exists():
        shutil.rmtree(pdir)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _list_projects() -> list[dict]:
    """Return metadata for each project under PROJECTS_ROOT."""
    if not PROJECTS_ROOT.is_dir():
        return []

    projects = []
    for d in sorted(PROJECTS_ROOT.iterdir()):
        meta_path = d / "project.json"
        if d.is_dir() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
            # Count images
            images_all = d / "images" / "all"
            image_count = (
                sum(1 for p in images_all.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
                if images_all.is_dir() else 0
            )
            projects.append({
                "name": d.name,
                "path": d,
                "base_model": meta.get("base_model", ""),
                "classes": meta.get("classes", []),
                "image_count": image_count,
            })
    return projects


def _delete_all_projects() -> None:
    """Remove all projects under PROJECTS_ROOT."""
    import shutil
    if PROJECTS_ROOT.is_dir():
        shutil.rmtree(PROJECTS_ROOT)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _project_selector() -> Path | None:
    """Show project selection screen. Returns project dir or None."""
    # Hero banner
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(str(_LOGO_PATH), width=150)
    with col_title:
        st.markdown(
            '<div style="padding:0.5rem 0;">'
            '<h1 style="margin:0 0 0.25rem 0;font-size:2rem;">'
            'PyZM Training Studio</h1>'
            '<p style="color:#94A3B8;margin:0;font-size:0.95rem;">'
            'Fine-tune YOLO models on your own data</p></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "If your ZoneMinder detections are wrong \u2014 missing objects, false alarms, "
        "or misidentified objects \u2014 this tool lets you teach the model using your "
        "own camera footage. No ML expertise needed."
    )
    st.divider()

    projects = _list_projects()

    if projects:
        st.markdown(":material/folder_open: **Your Projects**")
        for proj in projects:
            parts = []
            if proj["image_count"]:
                parts.append(f"&#x1F5BC; {proj['image_count']} images")
            if proj["base_model"]:
                parts.append(f"&#x1F916; {proj['base_model']}")
            if proj["classes"]:
                parts.append(f"&#x1F3F7; {len(proj['classes'])} classes")
            meta_str = " &nbsp;&middot;&nbsp; ".join(parts) if parts else "Empty project"

            col_card, col_open, col_del = st.columns([6, 1, 0.5])
            with col_card:
                st.markdown(
                    f'<div class="pyzm-project-card">'
                    f'<div class="proj-name">&#x1F4C1; {proj["name"]}</div>'
                    f'<span class="meta">{meta_str}</span></div>',
                    unsafe_allow_html=True,
                )
            with col_open:
                if st.button(":material/open_in_new: Open", key=f"open_{proj['name']}"):
                    st.session_state["project_name"] = proj["name"]
                    st.rerun()
            with col_del:
                if st.button(":material/delete:", key=f"del_{proj['name']}"):
                    st.session_state["_confirm_delete"] = proj["name"]
                    st.rerun()

        # Confirmation dialog for delete
        confirm_name = st.session_state.get("_confirm_delete")
        if confirm_name:
            @st.dialog("Delete project?")
            def _confirm_delete():
                st.warning(f"This will permanently delete **{confirm_name}** and all its data.")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Delete", type="primary", key="_confirm_del_yes"):
                        import shutil
                        shutil.rmtree(PROJECTS_ROOT / confirm_name, ignore_errors=True)
                        st.session_state.pop("_confirm_delete", None)
                        st.rerun()
                with col_no:
                    if st.button("Cancel", key="_confirm_del_no"):
                        st.session_state.pop("_confirm_delete", None)
                        st.rerun()
            _confirm_delete()

        st.divider()

    st.markdown(":material/add_circle: **Create New Project**")
    col_input, col_create = st.columns([3, 1])
    with col_input:
        new_name = st.text_input(
            "Project name",
            placeholder="e.g. license_plates",
            label_visibility="collapsed",
        )
    with col_create:
        create_clicked = st.button(":material/rocket_launch: Create", type="primary", width="stretch")

    if create_clicked:
        name = (new_name or "").strip()
        if not name:
            st.error("Enter a project name.")
            return None
        if len(name) > 100:
            st.error("Project name too long (max 100 characters).")
            return None
        # Sanitise: allow alphanumeric, dashes, underscores
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        if ".." in safe_name or "/" in safe_name:
            st.error("Invalid project name.")
            return None
        pdir = (PROJECTS_ROOT / safe_name).resolve()
        if not str(pdir).startswith(str(PROJECTS_ROOT.resolve())):
            st.error("Invalid project name.")
            return None
        if pdir.exists():
            st.error(f"Project '{safe_name}' already exists.")
            return None
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        st.session_state["project_name"] = safe_name
        st.rerun()

    # Delete all projects
    if projects:
        st.divider()
        # Inject red styling for the next button
        st.markdown(
            "<style>#delete-all-section button { "
            "background-color: #E74C3C !important; "
            "border-color: #C0392B !important; color: white !important; }"
            "</style>"
            "<div id='delete-all-section'>",
            unsafe_allow_html=True,
        )
        if st.button("Delete All Projects", key="delete_all_projects"):
            st.session_state["_confirm_delete_all"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.get("_confirm_delete_all"):
            st.error("This will permanently delete **all** projects and their data.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete all", type="primary", key="confirm_delete_all"):
                    _delete_all_projects()
            with c2:
                if st.button("Cancel", key="cancel_delete_all"):
                    st.session_state.pop("_confirm_delete_all", None)
                    st.rerun()

    return None


def _ensure_project(project_name: str) -> Path:
    """Ensure the project directory exists and return its path."""
    pdir = PROJECTS_ROOT / project_name
    if not (pdir / "project.json").exists():
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
    return pdir


def _model_picker(args: argparse.Namespace, pdir: Path) -> None:
    """Show model selection UI. Saves choice and moves to browse phase."""
    _section_header("&#x1F916;", "Starting Model")
    st.caption("Pick the model you currently use for detection. We'll improve it using your corrections.")

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
    )

    model_path = model_paths.get(base_model, "")
    model_classes = _read_model_classes(model_path)
    if model_classes:
        st.caption(f"This model detects **{len(model_classes)}** classes.")
    else:
        st.caption("Could not read classes from model metadata.")

    if st.button(":material/arrow_forward: Continue", type="primary"):
        meta_path = pdir / "project.json"
        meta = json.loads(meta_path.read_text())
        meta["base_model"] = base_model
        meta_path.write_text(json.dumps(meta, indent=2))

        st.session_state["base_model"] = base_model
        st.session_state["model_class_names"] = model_classes
        st.session_state["active_phase"] = "select"
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="pyZM Training Studio",
        page_icon=":material/model_training:",
        layout="wide",
    )
    _inject_css()

    args = _parse_app_args()

    # --- Project selection ---
    # If --workspace-dir is passed, skip project selector (backward compat)
    if args.workspace_dir:
        pdir = Path(args.workspace_dir)
        if not (pdir / "project.json").exists():
            ds = YOLODataset(project_dir=pdir, classes=[])
            ds.init_project()
        st.session_state["project_name"] = pdir.name
    elif not st.session_state.get("project_name"):
        _project_selector()
        return
    else:
        pdir = _ensure_project(st.session_state["project_name"])

    st.session_state["workspace_dir"] = str(pdir)
    ds = YOLODataset.load(pdir)
    store = VerificationStore(pdir)

    # First run: pick a base model before proceeding
    if not st.session_state.get("base_model"):
        # Try to load from saved metadata
        meta_path = pdir / "project.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("base_model"):
                st.session_state["base_model"] = meta["base_model"]
        if not st.session_state.get("base_model"):
            _sidebar(None, None)
            _model_picker(args, pdir)
            return

    # Load model class names into session if not present
    if "model_class_names" not in st.session_state:
        base_model = st.session_state.get("base_model", "yolo11s")
        available = _scan_models(args.base_path)
        model_paths = {m["name"]: m["path"] for m in available}
        model_path = model_paths.get(base_model, "")
        st.session_state["model_class_names"] = _read_model_classes(model_path)

    _seed_from_legacy_labels(ds, store)

    # Sidebar controls everything
    phase = _sidebar(ds, store)

    # Map stale "upload" phase to "select" (phase was removed)
    if phase == "upload":
        phase = "select"
        st.session_state["active_phase"] = "select"

    # Render active phase
    if phase == "select":
        _phase_select(ds, store, args)
    elif phase == "review":
        _phase_review(ds, store, args)
    elif phase == "train":
        _phase_train(ds, store, args)


if __name__ == "__main__":
    main()
