"""Streamlit UI for YOLO fine-tuning (problem-driven workflow).

Launch with::

    python -m pyzm.train
    streamlit run pyzm/train/app.py -- --base-path /path/to/models

Phases (sidebar-driven):
    1. Select Images -- import from YOLO datasets, raw images, or correct model detections
    2. Review Detections -- approve/edit/delete auto-detected objects
    3. Train & Export -- fine-tune and export ONNX
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from io import BytesIO
from pathlib import Path

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

from pyzm.train.dataset import YOLODataset
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
    /* Green check_circle icons in sidebar buttons and expanders */
    [data-testid="stExpander"] span[data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] span[data-testid="stIconMaterial"] {
        color: #27AE60 !important;
    }
    </style>""", unsafe_allow_html=True)


def _section_header(icon: str, title: str) -> None:
    """Render a styled section header with a material icon."""
    st.markdown(
        f'<div class="pyzm-section-header">'
        f'<span style="font-size:1.3rem;">{icon}</span>'
        f'<h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def _step_expander(
    label: str, *, done: bool = False, detail: str = "",
):
    """Collapsible step: collapsed with checkmark when done, expanded when pending."""
    prefix = ":material/check_circle:" if done else "\u25CB"
    suffix = f" \u2014 {detail}" if detail else ""
    return st.expander(f"{prefix} {label}{suffix}", expanded=not done)


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
            icon = ":material/check_circle:" if done else ""
            btn_label = f"{prefix}{icon} {label}" if icon else f"{prefix}{label}"
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
        st.markdown(
            "<style>#pyzm-delete-project button { "
            "background-color: #E74C3C !important; "
            "border-color: #C0392B !important; color: white !important; }"
            "</style>"
            "<div id='pyzm-delete-project'>",
            unsafe_allow_html=True,
        )
        if st.button(":material/delete: Delete Project", key="reset_workspace"):
            st.session_state["_confirm_reset"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.get("_confirm_reset"):
            st.warning(
                f"Permanently delete **{project_name}** and all its data?"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete", type="primary", key="reset_confirm"):
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
    # Configure logging so all pyzm messages (pyzm.train, pyzm.ml, etc.)
    # appear in the console
    _pyzm_logger = logging.getLogger("pyzm")
    if not _pyzm_logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        _pyzm_logger.addHandler(_handler)
        _pyzm_logger.setLevel(logging.DEBUG)

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

    # Restore base_model from saved project metadata
    if not st.session_state.get("base_model"):
        meta_path = pdir / "project.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("base_model"):
                st.session_state["base_model"] = meta["base_model"]

    # Load model class names into session if not present
    if "model_class_names" not in st.session_state:
        # Try project settings first (persisted), then read from model file
        saved_classes = ds.get_setting("model_class_names")
        if saved_classes:
            st.session_state["model_class_names"] = saved_classes
        elif st.session_state.get("base_model"):
            base_model = st.session_state["base_model"]
            available = _scan_models(args.base_path)
            model_paths = {m["name"]: m["path"] for m in available}
            model_path = model_paths.get(base_model, "")
            classes_from_model = _read_model_classes(model_path)
            st.session_state["model_class_names"] = classes_from_model
            if classes_from_model:
                ds.set_setting("model_class_names", classes_from_model)

    _seed_from_legacy_labels(ds, store)

    # Sidebar controls everything
    phase = _sidebar(ds, store)

    # Map stale "upload" phase to "select" (phase was removed)
    if phase == "upload":
        phase = "select"
        st.session_state["active_phase"] = "select"

    # Render active phase (late imports to avoid circular dependency)
    if phase == "select":
        from pyzm.train._import_panel import _phase_select
        _phase_select(ds, store, args)
    elif phase == "review":
        from pyzm.train._review_panel import _phase_review
        _phase_review(ds, store, args)
    elif phase == "train":
        from pyzm.train._train_panel import _phase_train
        _phase_train(ds, store, args)


if __name__ == "__main__":
    main()
