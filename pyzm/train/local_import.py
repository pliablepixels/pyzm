"""Local dataset import for the training UI.

Supports three modes:

1. **Pre-annotated YOLO dataset** — folder with data.yaml + images/ + labels/,
   imported with all annotations pre-approved.
2. **Raw images** — unannotated images from a local folder, auto-detected and
   imported with ``fully_reviewed=False`` so they go through the Review phase.
3. **Correct Model Detections** — batch-detect on import so the user can
   immediately correct model mistakes in the Review phase.
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


# ===================================================================
# Auto-detect (pure Python — no Streamlit dependency)
# ===================================================================

def auto_detect_image(
    image_path: Path,
    *,
    base_model: str = "yolo11s",
    workspace_dir: str | Path | None = None,
    base_path: str = "/var/lib/zmeventnotification/models",
    processor: str = "gpu",
    model_classes: list[str] | None = None,
    min_confidence: float = 0.3,
) -> list[VerifiedDetection]:
    """Run auto-detect on a single image and return PENDING VerifiedDetections.

    This is a pure-Python function with no Streamlit dependency, so it can
    be reused by both the UI and the headless CLI.

    Parameters
    ----------
    image_path:
        Path to the image file.
    base_model:
        Name of the base YOLO model (e.g. ``"yolo11s"``).
    workspace_dir:
        Project workspace directory. If a fine-tuned ``best.onnx`` exists
        inside ``<workspace_dir>/runs/train/weights/``, it is used instead
        of *base_model*.
    base_path:
        Base path passed to :class:`pyzm.ml.detector.Detector`.
    processor:
        ``"gpu"`` or ``"cpu"``.
    model_classes:
        Class names the model can detect. If empty *and* no fine-tuned
        model exists, detection is skipped.
    min_confidence:
        Minimum confidence threshold. Detections below this are discarded.
        Default: ``0.3``.
    """
    # Prefer the exported ONNX (Detector doesn't handle .pt files).
    # Fall back to best.pt only as an existence check — if ONNX is
    # missing but .pt exists, the model was trained but not exported.
    weights_dir = (
        Path(workspace_dir) / "runs" / "train" / "weights"
        if workspace_dir else None
    )
    best_onnx = weights_dir / "best.onnx" if weights_dir else None
    best_pt = weights_dir / "best.pt" if weights_dir else None
    has_trained = (best_onnx is not None and best_onnx.exists())

    detections: list[VerifiedDetection] = []
    if not ((model_classes or []) or has_trained):
        return detections

    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return detections
        h, w = img.shape[:2]

        from pyzm.ml.detector import Detector

        model_to_use = str(best_onnx) if has_trained else base_model
        logger.info(
            "Auto-detect using model: %s (min_confidence=%.0f%%, processor=%s)",
            model_to_use, min_confidence * 100, processor,
        )
        det = Detector(
            models=[model_to_use],
            base_path=base_path,
            processor=processor,
        )
        result = det.detect(img)
        det_idx = 0
        for d in result.detections:
            conf = getattr(d, "confidence", None)
            if conf is not None and conf < min_confidence:
                continue
            b = d.bbox
            cx = ((b.x1 + b.x2) / 2) / w
            cy = ((b.y1 + b.y2) / 2) / h
            bw = (b.x2 - b.x1) / w
            bh = (b.y2 - b.y1) / h
            ann = Annotation(class_id=0, cx=cx, cy=cy, w=bw, h=bh)
            detections.append(VerifiedDetection(
                detection_id=f"det_{det_idx}",
                original=ann,
                status=DetectionStatus.PENDING,
                original_label=d.label,
                confidence=conf,
            ))
            det_idx += 1

        if detections:
            summary = ", ".join(
                f"{d.original_label}:{d.confidence:.0%}"
                if d.confidence else d.original_label
                for d in detections
            )
            logger.info("Auto-detect %s: %s", image_path.name, summary)
        else:
            logger.info("Auto-detect %s: no detections", image_path.name)
    except Exception as exc:
        logger.warning("Auto-detect failed for %s: %s", image_path.name, exc)

    return detections


# ===================================================================
# Batch import + detect (Correct Model Detections)
# ===================================================================

def import_correct_model(
    ds: YOLODataset,
    store: VerificationStore,
    folder: Path,
    *,
    base_model: str = "yolo11s",
    workspace_dir: str | Path | None = None,
    base_path: str = "/var/lib/zmeventnotification/models",
    processor: str = "gpu",
    model_classes: list[str] | None = None,
    min_confidence: float = 0.3,
    max_images: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Import images from *folder* and run detection on each one.

    Unlike :func:`import_raw_images`, detections are stored as PENDING
    immediately at import time so the user can correct them in the Review
    phase without having to trigger auto-detect per image.

    Returns ``(image_count, detection_count)``.
    """
    all_images = _find_images(folder)
    if not all_images:
        return 0, 0

    if max_images > 0 and len(all_images) > max_images:
        import random
        all_images = random.sample(all_images, max_images)

    img_count = 0
    det_count = 0
    total = len(all_images)

    for i, img_path in enumerate(all_images):
        dest = ds.add_image(img_path, [])
        img_count += 1

        detections = auto_detect_image(
            dest,
            base_model=base_model,
            workspace_dir=workspace_dir,
            base_path=base_path,
            processor=processor,
            model_classes=model_classes,
            min_confidence=min_confidence,
        )
        det_count += len(detections)

        store.set(ImageVerification(
            image_name=dest.name,
            detections=detections,
            fully_reviewed=False,
            detected=True,
        ))

        if progress_callback:
            summary = _detection_summary(detections)
            progress_callback(i + 1, total, dest.name, summary)

    store.save()
    ds.set_setting("data_source", "correct_model")
    return img_count, det_count


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
            detected=True,
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
        ds.set_setting("data_source", "yolo_dataset")
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
        from pyzm.train._import_panel import _upload_panel
        _upload_panel(
            ds, store, args,
            label="Upload images that we will use to train",
            data_source="raw_images",
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
                            ds.set_setting("data_source", "raw_images")
                            st.session_state.pop(scan_key, None)
                            st.toast(f"Imported {img_count} images")
                            st.rerun()


# ===================================================================
# Manually Annotate (merged panel)
# ===================================================================

def manual_annotate_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
) -> None:
    """Unified panel for importing images with optional auto-detection."""
    import streamlit as st

    auto_detect = st.checkbox(
        "Auto-detect objects at import",
        value=True,
        help=(
            "Run the base model on each image during import to give you a "
            "starting point. You can always run detection later in the Review phase."
        ),
        key="_manual_auto_detect",
    )

    min_confidence = 0.3
    if auto_detect:
        min_confidence = st.slider(
            "Min confidence",
            min_value=0.1, max_value=0.9, value=0.3, step=0.05,
            help="Detections below this confidence are discarded.",
            key="_manual_min_confidence",
        )

    method = st.radio(
        "Import method",
        ["Upload Images", "Server Folder"],
        horizontal=True,
        key="manual_import_method",
    )

    if method == "Upload Images":
        if auto_detect:
            _correct_upload_panel(ds, store, args, min_confidence=min_confidence)
        else:
            from pyzm.train._import_panel import _upload_panel
            _upload_panel(
                ds, store, args,
                label="Upload images to annotate",
                data_source="raw_images",
            )
    else:
        # Restore saved path when reopening an existing project
        if "_manual_folder_path" not in st.session_state:
            saved = ds.get_setting("import_source_path", "")
            if saved:
                st.session_state["_manual_folder_path"] = saved

        folder_path = st.text_input(
            "Path to image folder",
            placeholder="/path/to/my_images",
            help="Folder containing image files (.jpg, .jpeg, .png, .bmp, .webp).",
            key="_manual_folder_path",
        )
        _folder_picker("_manual_folder_path", label="Browse for folder")

        if not folder_path or not folder_path.strip():
            st.info(
                "Enter the path to a folder of images. "
                "You'll review and correct detections in the next phase."
            )
        else:
            folder = Path(folder_path.strip())
            scan_key = "_manual_scan_result"

            col_scan, col_import = st.columns(2)
            with col_scan:
                if st.button("Scan", key="manual_scan"):
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
                    limit_all = st.checkbox("Import all images", value=True, key="_manual_import_all")
                    max_images = 0
                    if not limit_all:
                        max_images = st.number_input(
                            "Max images to import",
                            min_value=1, max_value=total, value=min(100, total), step=10,
                            key="_manual_max_images",
                        )
                    btn_label = "Import & Detect" if auto_detect else "Import"
                    with col_import:
                        if st.button(btn_label, type="primary", key="manual_import"):
                            if auto_detect:
                                _manual_folder_import_detect(
                                    ds, store, args, Path(scan["folder"]),
                                    max_images=max_images,
                                    min_confidence=min_confidence,
                                )
                            else:
                                img_count = _import_raw_images(
                                    ds, store, Path(scan["folder"]),
                                    max_images=max_images,
                                )
                                ds.set_setting("import_source_path", scan["folder"])
                                ds.set_setting("data_source", "raw_images")
                                st.toast(f"Imported {img_count} images")
                            st.session_state.pop(scan_key, None)
                            st.rerun()


def _manual_folder_import_detect(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
    folder: Path,
    max_images: int = 0,
    min_confidence: float = 0.3,
) -> None:
    """Import images from a folder and run detection on each."""
    import streamlit as st

    base_model = st.session_state.get("base_model", "yolo11s")
    pdir = st.session_state.get("workspace_dir")
    model_classes = st.session_state.get("model_class_names", [])

    progress = st.progress(0, text="Importing & detecting...")
    log_area = st.empty()
    log_lines: list[str] = []

    def _cb(current: int, total: int, name: str = "", summary: str = "") -> None:
        progress.progress(current / total, text=f"Importing & detecting... {current}/{total}")
        if name:
            log_lines.append(f"{name}: {summary}")
            log_area.code("\n".join(log_lines[-8:]), language=None)

    img_count, det_count = import_correct_model(
        ds, store, folder,
        base_model=base_model,
        workspace_dir=pdir,
        base_path=args.base_path,
        processor=args.processor,
        model_classes=model_classes,
        min_confidence=min_confidence,
        max_images=max_images,
        progress_callback=_cb,
    )
    progress.progress(1.0, text=f"Imported {img_count} images, {det_count} detections")
    ds.set_setting("import_source_path", str(folder))
    st.toast(f"Imported {img_count} images with {det_count} detections")


# ===================================================================
# Correct Model Detections (UI panel -- kept for headless/legacy use)
# ===================================================================

def _detection_summary(detections: list[VerifiedDetection]) -> str:
    """One-line summary of detections for progress display."""
    if not detections:
        return "no detections"
    parts = []
    for d in detections:
        if d.confidence:
            parts.append(f"{d.original_label}:{d.confidence:.0%}")
        else:
            parts.append(d.original_label)
    return ", ".join(parts)


def _correct_import_and_detect(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
    images: list[Path],
    min_confidence: float = 0.3,
) -> None:
    """Run detection on a list of already-imported image paths and store results."""
    import streamlit as st

    base_model = st.session_state.get("base_model", "yolo11s")
    pdir = st.session_state.get("workspace_dir")
    model_classes = st.session_state.get("model_class_names", [])

    progress = st.progress(0, text="Detecting...")
    log_area = st.empty()
    det_count = 0
    total = len(images)
    log_lines: list[str] = []

    for i, dest in enumerate(images):
        detections = auto_detect_image(
            dest,
            base_model=base_model,
            workspace_dir=pdir,
            base_path=args.base_path,
            processor=args.processor,
            model_classes=model_classes,
            min_confidence=min_confidence,
        )
        det_count += len(detections)

        store.set(ImageVerification(
            image_name=dest.name,
            detections=detections,
            fully_reviewed=False,
            detected=True,
        ))

        summary = _detection_summary(detections)
        log_lines.append(f"{dest.name}: {summary}")
        # Show last 8 lines so user can see progress
        log_area.code("\n".join(log_lines[-8:]), language=None)
        progress.progress((i + 1) / total, text=f"Detecting... {i + 1}/{total}")

    store.save()
    ds.set_setting("data_source", "correct_model")
    progress.progress(1.0, text=f"Done: {det_count} detections across {total} images")
    st.toast(f"Detected {det_count} objects across {total} images")


def _correct_upload_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
    min_confidence: float = 0.3,
) -> None:
    """Upload images and immediately run detection on them."""
    import streamlit as st
    import tempfile

    upload_key = st.session_state.get("_correct_upload_key", 0)
    uploaded = st.file_uploader(
        "Upload images where detection failed or needs improvement",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=f"correct_uploader_{upload_key}",
    )
    if not uploaded:
        return

    # Save all images to disk
    import_bar = st.progress(0, text="Importing images...")
    destinations: list[Path] = []
    for i, f in enumerate(uploaded):
        tmp = Path(tempfile.mkdtemp()) / f.name
        tmp.write_bytes(f.read())
        destinations.append(ds.add_image(tmp, []))
        import_bar.progress((i + 1) / len(uploaded), text=f"Importing {i + 1}/{len(uploaded)}")
    import_bar.empty()

    # Run detection on all imported images
    _correct_import_and_detect(ds, store, args, destinations, min_confidence=min_confidence)

    st.session_state["_correct_upload_key"] = upload_key + 1
    st.rerun()


def correct_model_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
) -> None:
    """Streamlit panel for the 'Correct Model Detections' import source.

    Lets the user upload images or point at a server folder, runs batch
    detection on import, and stores results as PENDING so they can be
    corrected in the Review phase.
    """
    import streamlit as st

    # Show which model will be used
    base_model = st.session_state.get("base_model", "yolo11s")
    pdir = st.session_state.get("workspace_dir")
    best_onnx = Path(pdir) / "runs" / "train" / "weights" / "best.onnx" if pdir else None
    has_trained = best_onnx is not None and best_onnx.exists()
    model_display = str(best_onnx) if has_trained else base_model
    st.caption(f"Detection model: **{model_display}**")

    min_confidence = st.slider(
        "Min confidence",
        min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        help="Detections below this confidence are discarded.",
        key="_correct_min_confidence",
    )

    method = st.radio(
        "Import method",
        ["Upload Images", "Server Folder"],
        horizontal=True,
        key="correct_import_method",
    )

    if method == "Upload Images":
        _correct_upload_panel(ds, store, args, min_confidence=min_confidence)
    else:
        # Restore saved path when reopening an existing project
        if "_correct_folder_path" not in st.session_state:
            saved = ds.get_setting("import_source_path", "")
            if saved:
                st.session_state["_correct_folder_path"] = saved

        folder_path = st.text_input(
            "Path to image folder",
            placeholder="/path/to/images",
            help="Folder containing images where you suspect the model is wrong.",
            key="_correct_folder_path",
        )
        _folder_picker("_correct_folder_path", label="Browse for folder")

        if not folder_path or not folder_path.strip():
            st.info(
                "Point at a folder of images where you think your model is "
                "making mistakes. We'll run detection on every image and let "
                "you correct the results."
            )
            return

        folder = Path(folder_path.strip())
        scan_key = "_correct_scan_result"

        col_scan, col_import = st.columns(2)
        with col_scan:
            if st.button("Scan", key="correct_scan"):
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
                with col_import:
                    if st.button("Import & Detect", type="primary", key="correct_import"):
                        progress = st.progress(0, text="Importing & detecting...")
                        log_area = st.empty()
                        _log_lines: list[str] = []

                        def _cb(
                            current: int,
                            total_: int,
                            name: str = "",
                            summary: str = "",
                        ) -> None:
                            progress.progress(
                                current / total_,
                                text=f"Importing & detecting... {current}/{total_}",
                            )
                            if name:
                                _log_lines.append(f"{name}: {summary}")
                                log_area.code(
                                    "\n".join(_log_lines[-8:]), language=None,
                                )

                        model_classes = st.session_state.get("model_class_names", [])
                        img_count, det_count = import_correct_model(
                            ds, store, Path(scan["folder"]),
                            base_model=base_model,
                            workspace_dir=pdir,
                            base_path=args.base_path,
                            processor=args.processor,
                            model_classes=model_classes,
                            min_confidence=min_confidence,
                            progress_callback=_cb,
                        )
                        progress.progress(
                            1.0,
                            text=f"Imported {img_count} images, {det_count} detections",
                        )
                        ds.set_setting("import_source_path", scan["folder"])
                        st.session_state.pop(scan_key, None)
                        st.toast(
                            f"Imported {img_count} images with {det_count} detections"
                        )
                        st.rerun()
