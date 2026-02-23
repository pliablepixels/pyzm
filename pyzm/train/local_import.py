"""Local dataset import for the training UI.

Supports two modes:

1. **Pre-annotated YOLO dataset** — folder with data.yaml + images/ + labels/,
   imported with all annotations pre-approved.
2. **Raw images** — unannotated images from a local folder, auto-detected and
   imported with ``fully_reviewed=False`` so they go through the Review phase.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _folder_picker(session_key: str, label: str = "Browse") -> None:
    """Inline folder navigator with cascading selectboxes."""
    import streamlit as st

    nav_key = f"_nav_{session_key}"

    # Sync: if the user typed a valid directory in the text input, start there
    typed = st.session_state.get(session_key, "").strip()
    if typed and Path(typed).is_dir():
        st.session_state[nav_key] = typed
    elif nav_key not in st.session_state:
        st.session_state[nav_key] = str(Path.home())

    def _list_dirs(path: Path) -> list[str]:
        try:
            return sorted(
                p.name for p in path.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        except PermissionError:
            return []

    current = Path(st.session_state[nav_key])

    with st.expander(label, expanded=False):
        st.caption(f"`{current}`")

        col_sel, col_ok = st.columns([4, 1])

        # Go up
        with col_ok:
            can_go_up = current != current.parent
            if st.button(":material/drive_folder_upload:", key=f"{nav_key}_up", disabled=not can_go_up):
                st.session_state[nav_key] = str(current.parent)
                st.rerun()

        # Subfolder dropdown
        subdirs = _list_dirs(current)
        if subdirs:
            with col_sel:
                picked = st.selectbox(
                    "Enter subfolder",
                    options=[""] + subdirs,
                    format_func=lambda x: x or "-- select subfolder --",
                    label_visibility="collapsed",
                    key=f"{nav_key}_sub",
                )
                if picked:
                    st.session_state[nav_key] = str(current / picked)
                    st.session_state.pop(f"{nav_key}_sub", None)
                    st.rerun()

        def _select():
            st.session_state[session_key] = st.session_state[nav_key]

        st.button("Select", type="primary", key=f"{nav_key}_confirm",
                  on_click=_select)


def validate_yolo_folder(folder: Path) -> dict | str:
    """Validate a local YOLO dataset folder.

    Returns the parsed data.yaml dict on success, or an error string.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return f"Not a directory: {folder}"

    yaml_path = folder / "data.yaml"
    if not yaml_path.exists():
        return f"Missing data.yaml in {folder}"

    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as exc:
        return f"Failed to parse data.yaml: {exc}"

    if not isinstance(data, dict):
        return "data.yaml is not a valid YAML mapping"

    if "names" not in data:
        return "data.yaml missing required 'names' key"

    names = data["names"]
    if isinstance(names, list):
        data["names"] = {i: n for i, n in enumerate(names)}
    elif not isinstance(names, dict):
        return "'names' in data.yaml must be a list or dict"

    # Standard layout: top-level images/ + labels/
    images_dir = folder / "images"
    labels_dir = folder / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        data["_splits"] = [(images_dir, labels_dir)]
        return data

    # Roboflow-style split dirs: train/images/ + train/labels/, etc.
    splits: list[tuple[Path, Path]] = []
    for name in ("train", "valid", "val", "test"):
        si = folder / name / "images"
        sl = folder / name / "labels"
        if si.is_dir() and sl.is_dir():
            splits.append((si, sl))
    if splits:
        data["_splits"] = splits
        return data

    return f"No images/labels directories found in {folder}"


def _find_images(images_dir: Path) -> list[Path]:
    """Recursively find all image files under a directory."""
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )


def _find_matching_label(image_path: Path, images_dir: Path, labels_dir: Path) -> Path | None:
    """Find the label .txt file that corresponds to an image.

    Mirrors the relative path from images/ to labels/, swapping extension.
    """
    rel = image_path.relative_to(images_dir)
    label_path = labels_dir / rel.with_suffix(".txt")
    return label_path if label_path.exists() else None


def _infer_split(img_path: Path, images_dir: Path) -> str | None:
    """Infer the original train/val assignment from the directory layout.

    Returns ``"train"``, ``"val"``, or ``None`` (unknown).

    Supports:
    - Roboflow: ``train/images/img.jpg``, ``valid/images/img.jpg``
    - Standard subdirs: ``images/train/img.jpg``, ``images/val/img.jpg``
    - Flat layout (``images/img.jpg``) → ``None``
    """
    # Roboflow: images_dir is e.g. <root>/train/images — parent is the split
    parent_name = images_dir.parent.name.lower()
    if parent_name in ("train",):
        return "train"
    if parent_name in ("valid", "val"):
        return "val"

    # Standard subdirs: images_dir is e.g. <root>/images,
    # image might be at images/train/img.jpg
    try:
        rel = img_path.relative_to(images_dir)
    except ValueError:
        return None
    if rel.parts and len(rel.parts) > 1:
        first = rel.parts[0].lower()
        if first in ("train",):
            return "train"
        if first in ("valid", "val"):
            return "val"

    return None


def _select_by_class(
    all_files: list[tuple[Path, Path, Path]],
    max_per_class: int,
) -> list[tuple[Path, Path, Path]]:
    """Return a subset of *all_files* with at most *max_per_class* images per class.

    Scans each label file to discover which class IDs it contains, then
    keeps the first *max_per_class* images for every class.  Because an
    image can contain multiple classes, the final set is de-duplicated.
    """
    from collections import defaultdict

    class_to_files: dict[int, list[int]] = defaultdict(list)

    for idx, (img_path, images_dir, labels_dir) in enumerate(all_files):
        label_path = _find_matching_label(img_path, images_dir, labels_dir)
        if not label_path:
            continue
        seen_classes: set[int] = set()
        for line in label_path.read_text().strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                class_id = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if class_id not in seen_classes:
                seen_classes.add(class_id)
                class_to_files[class_id].append(idx)

    selected: set[int] = set()
    for class_id in sorted(class_to_files):
        for idx in class_to_files[class_id][:max_per_class]:
            selected.add(idx)

    return [all_files[i] for i in sorted(selected)]


def import_local_dataset(
    ds: YOLODataset,
    store: VerificationStore,
    splits: list[tuple[Path, Path]],
    names_map: dict[int, str],
    max_images: int = 0,
    max_per_class: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Import images and labels from one or more (images_dir, labels_dir) pairs.

    Annotations are written into the dataset at import time (no empty
    label files).  The original train/val split is preserved in
    ``split_map`` so that :meth:`YOLODataset.split` can honor it.

    Parameters
    ----------
    max_images:
        If > 0, import at most this many images (randomly sampled).
    max_per_class:
        If > 0, keep at most this many images per class ID before
        applying *max_images*.  Images with multiple classes count
        toward every class they contain.
    progress_callback:
        Optional ``(current, total)`` callback invoked after each image.

    Returns (image_count, detection_count).
    """
    # Collect all (image_path, images_dir, labels_dir) across splits
    all_files: list[tuple[Path, Path, Path]] = []
    for images_dir, labels_dir in splits:
        for img_path in _find_images(images_dir):
            all_files.append((img_path, images_dir, labels_dir))

    if not all_files:
        return 0, 0

    if max_per_class > 0:
        all_files = _select_by_class(all_files, max_per_class)

    if max_images > 0 and len(all_files) > max_images:
        import random
        all_files = random.sample(all_files, max_images)

    # Remap class IDs to contiguous 0-based indices (needed when classes
    # are ignored/filtered, leaving gaps like {1: "gun"} → {0: "gun"}).
    id_remap = {old_id: new_id for new_id, old_id in enumerate(sorted(names_map))}

    img_count = 0
    det_count = 0
    split_map: dict[str, str] = {}
    total = len(all_files)

    for i, (img_path, images_dir, labels_dir) in enumerate(all_files):
        # Parse annotations BEFORE add_image so labels are populated on import
        annotations: list[Annotation] = []
        detections: list[VerifiedDetection] = []
        label_path = _find_matching_label(img_path, images_dir, labels_dir)
        if label_path:
            lines = label_path.read_text().strip().splitlines()
            for j, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = Annotation.from_yolo_line(line)
                except (ValueError, IndexError):
                    logger.warning("Skipping malformed line in %s: %s", label_path, line)
                    continue
                class_name = names_map.get(ann.class_id)
                if class_name is None:
                    continue  # Skip annotations for ignored/unknown classes
                ann = Annotation(
                    class_id=id_remap[ann.class_id],
                    cx=ann.cx, cy=ann.cy, w=ann.w, h=ann.h,
                )
                annotations.append(ann)
                detections.append(VerifiedDetection(
                    detection_id=f"det_{j}",
                    original=ann,
                    status=DetectionStatus.APPROVED,
                    original_label=class_name,
                ))
                det_count += 1

        dest = ds.add_image(img_path, annotations)
        img_count += 1

        # Record original split assignment
        orig_split = _infer_split(img_path, images_dir)
        if orig_split:
            split_map[dest.name] = orig_split

        iv = ImageVerification(
            image_name=dest.name,
            detections=detections,
            fully_reviewed=True,
        )
        store.set(iv)
        if progress_callback:
            progress_callback(i + 1, total)

    # Persist split map and import classes
    if split_map:
        ds.set_setting("split_map", split_map)
    ds.set_setting("import_classes", sorted(names_map.values()))

    store.save()
    return img_count, det_count


def _import_local_dataset(
    ds: YOLODataset,
    store: VerificationStore,
    splits: list[tuple[Path, Path]],
    names_map: dict[int, str],
    max_images: int = 0,
) -> tuple[int, int]:
    """Streamlit wrapper around :func:`import_local_dataset`."""
    import streamlit as st

    progress = st.progress(0, text="Importing local dataset...")

    def _cb(current: int, total: int) -> None:
        progress.progress(current / total, text=f"Importing... {current}/{total}")

    img_count, det_count = import_local_dataset(
        ds, store, splits, names_map, max_images=max_images, progress_callback=_cb,
    )
    progress.progress(1.0, text=f"Imported {img_count} images, {det_count} annotations")
    return img_count, det_count


def local_dataset_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args,
) -> None:
    """Streamlit panel for importing a local YOLO dataset."""
    import streamlit as st

    # Restore saved path when reopening an existing project
    if "_yolo_folder_path" not in st.session_state:
        saved = ds.get_setting("import_source_path", "")
        if saved:
            st.session_state["_yolo_folder_path"] = saved

    folder_path = st.text_input(
        "Path to YOLO dataset folder",
        placeholder="/path/to/my_dataset",
        help=(
            "Folder must contain data.yaml plus images/ and labels/ dirs. "
            "Standard (images/ + labels/) and Roboflow-style split "
            "(train/images/ + train/labels/, etc.) layouts are supported."
        ),
        key="_yolo_folder_path",
    )
    _folder_picker("_yolo_folder_path", label="Browse for folder")

    if not folder_path or not folder_path.strip():
        st.info(
            "Enter the path to a folder in standard YOLO format:\n"
            "```\n"
            "my_dataset/\n"
            "  data.yaml\n"
            "  images/\n"
            "  labels/\n"
            "```"
        )
        return

    folder = Path(folder_path.strip())

    # Auto-validate as soon as a folder path is entered
    result = validate_yolo_folder(folder)
    if isinstance(result, str):
        st.error(result)
        return

    names_map = result["names"]
    splits = result["_splits"]
    image_count = 0
    label_count = 0
    for images_dir, labels_dir in splits:
        imgs = _find_images(images_dir)
        image_count += len(imgs)
        label_count += sum(
            1 for img in imgs
            if _find_matching_label(img, images_dir, labels_dir) is not None
        )

    class_names = [names_map[k] for k in sorted(names_map)]
    st.success(
        f"Valid YOLO dataset: **{image_count}** images, "
        f"**{label_count}** label files, "
        f"**{len(class_names)}** classes"
    )
    st.caption(f"Classes: {', '.join(class_names)}")

    ignore_classes = st.multiselect(
        "Classes to ignore",
        options=class_names,
        default=[],
        help="Select classes to exclude from import. Their annotations will be skipped.",
        key="_ignore_classes",
    )
    if ignore_classes:
        names_map = {k: v for k, v in names_map.items() if v not in ignore_classes}
        kept = [names_map[k] for k in sorted(names_map)]
        st.info(f"Importing **{len(kept)}** classes: {', '.join(kept)}")

    limit_all = st.checkbox("Import all images", value=True, key="_local_import_all")
    max_images = 0
    if not limit_all:
        max_images = st.number_input(
            "Max images to import",
            min_value=1, max_value=image_count, value=min(100, image_count), step=10,
            key="_local_max_images",
        )

    if st.button("Import", type="primary", key="local_import"):
        img_count, det_count = _import_local_dataset(
            ds, store, splits, names_map, max_images=max_images,
        )
        ds.set_setting("import_source_path", folder_path)
        st.toast(f"Imported {img_count} images with {det_count} annotations")
        st.rerun()


# ===================================================================
# Raw (unannotated) image import
# ===================================================================

def import_raw_images(
    ds: YOLODataset,
    store: VerificationStore,
    folder: Path,
    max_images: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Import unannotated images from *folder*.

    Detection is deferred to the review phase.

    Parameters
    ----------
    progress_callback:
        Optional ``(current, total)`` callback invoked after each image.

    Returns the number of imported images.
    """
    all_images = _find_images(folder)
    if not all_images:
        return 0

    if max_images > 0 and len(all_images) > max_images:
        import random
        all_images = random.sample(all_images, max_images)

    img_count = 0
    total = len(all_images)

    for i, img_path in enumerate(all_images):
        dest = ds.add_image(img_path, [])
        img_count += 1

        store.set(ImageVerification(
            image_name=dest.name,
            detections=[],
            fully_reviewed=False,
        ))
        if progress_callback:
            progress_callback(i + 1, total)

    store.save()
    return img_count


def _import_raw_images(
    ds: YOLODataset,
    store: VerificationStore,
    folder: Path,
    max_images: int = 0,
) -> int:
    """Streamlit wrapper around :func:`import_raw_images`."""
    import streamlit as st

    progress = st.progress(0, text="Importing raw images...")

    def _cb(current: int, total: int) -> None:
        progress.progress(current / total, text=f"Importing... {current}/{total}")

    img_count = import_raw_images(
        ds, store, folder, max_images=max_images, progress_callback=_cb,
    )
    progress.progress(1.0, text=f"Imported {img_count} images")
    return img_count


def raw_images_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
) -> None:
    """Streamlit panel for importing raw (unannotated) images."""
    import streamlit as st

    method = st.radio(
        "Import method",
        ["Upload Images", "Server Folder"],
        horizontal=True,
        key="raw_import_method",
    )

    if method == "Upload Images":
        from pyzm.train.app import _upload_panel
        _upload_panel(
            ds, store, args,
            label="Upload images that we will use to train",
        )
    else:
        # Restore saved path when reopening an existing project
        if "_raw_folder_path" not in st.session_state:
            saved = ds.get_setting("import_source_path", "")
            if saved:
                st.session_state["_raw_folder_path"] = saved

        folder_path = st.text_input(
            "Path to image folder",
            placeholder="/path/to/my_images",
            help="Folder containing image files (.jpg, .jpeg, .png, .bmp, .webp).",
            key="_raw_folder_path",
        )
        _folder_picker("_raw_folder_path", label="Browse for folder")

        if not folder_path or not folder_path.strip():
            st.info(
                "Enter the path to a folder of images. "
                "No labels or data.yaml needed — the model will auto-detect objects "
                "and you can review/correct in the next phase."
            )
        else:
            folder = Path(folder_path.strip())
            scan_key = "_raw_scan_result"

            col_scan, col_import = st.columns(2)
            with col_scan:
                if st.button("Scan", key="raw_scan"):
                    if not folder.is_dir():
                        st.session_state[scan_key] = {"error": f"Not a directory: {folder}"}
                    else:
                        found = _find_images(folder)
                        st.session_state[scan_key] = {
                            "folder": str(folder),
                            "count": len(found),
                        }

            scan = st.session_state.get(scan_key)
            if scan:
                if "error" in scan:
                    st.error(scan["error"])
                elif scan["count"] == 0:
                    st.warning(f"No image files found in `{scan['folder']}`.")
                else:
                    total = scan["count"]
                    st.success(f"Found **{total}** images in `{scan['folder']}`.")
                    limit_all = st.checkbox("Import all images", value=True, key="_raw_import_all")
                    raw_max = 0
                    if not limit_all:
                        raw_max = st.number_input(
                            "Max images to import",
                            min_value=1, max_value=total, value=min(100, total), step=10,
                            key="_raw_max_images",
                        )
                    with col_import:
                        if st.button("Import", type="primary", key="raw_import"):
                            img_count = _import_raw_images(
                                ds, store, Path(scan["folder"]),
                                max_images=raw_max,
                            )
                            ds.set_setting("import_source_path", scan["folder"])
                            st.session_state.pop(scan_key, None)
                            st.toast(f"Imported {img_count} images")
                            st.rerun()
