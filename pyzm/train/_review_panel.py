"""Phase 2: Review Detections -- approve/edit/delete auto-detected objects."""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from pyzm.train._import_panel import _auto_detect_image
from pyzm.train.app import (
    MIN_IMAGES_PER_CLASS,
    _STATUS_COLORS,
    _draw_verified_image,
    _filtered_images,
    _generate_thumbnail_uri,
    _invalidate_thumbnail,
    _load_image_pil,
    _pagination_controls,
    _section_header,
)
from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)
from st_clickable_images import clickable_images


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
                format_func=lambda x: x or "\u2014",
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
                        format_func=lambda x: x or "\u2014",
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
