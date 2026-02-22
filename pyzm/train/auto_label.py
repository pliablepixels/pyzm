"""Auto-labeling using existing pyzm Detector.

Runs a single pre-trained YOLO model on uploaded images to pre-label known
classes, saving users from having to manually annotate every object.
Supports class mapping (e.g. car/truck/bus -> vehicle).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pyzm.train.dataset import Annotation

if TYPE_CHECKING:
    import numpy as np
    from pyzm.models.detection import DetectionResult

logger = logging.getLogger("pyzm.train")


def _bbox_to_annotation(
    x1: int, y1: int, x2: int, y2: int,
    class_id: int,
    img_w: int, img_h: int,
) -> Annotation:
    """Convert pixel bbox to YOLO normalised annotation."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return Annotation(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def detections_to_annotations(
    result: "DetectionResult",
    target_classes: list[str],
    img_w: int,
    img_h: int,
    class_mapping: dict[str, str] | None = None,
    min_confidence: float = 0.0,
) -> list[Annotation]:
    """Convert DetectionResult detections to Annotation list.

    Parameters
    ----------
    result:
        Detection output from pyzm Detector.
    target_classes:
        Final ordered class names (group names). Index = class_id.
    img_w, img_h:
        Image dimensions for normalisation.
    class_mapping:
        Optional dict mapping source label -> target label, e.g.
        ``{"car": "vehicle", "truck": "vehicle"}``.  If ``None``,
        detections are matched directly against *target_classes*.
    min_confidence:
        Skip detections below this confidence threshold.
    """
    annotations: list[Annotation] = []
    target_map = {name.lower(): idx for idx, name in enumerate(target_classes)}

    for det in result.detections:
        if det.confidence < min_confidence:
            continue
        source = det.label.lower()

        # Map source label to target group name
        if class_mapping:
            target = class_mapping.get(source)
            if target is None:
                continue
            target = target.lower()
        else:
            target = source

        if target not in target_map:
            continue

        b = det.bbox
        ann = _bbox_to_annotation(
            b.x1, b.y1, b.x2, b.y2,
            class_id=target_map[target],
            img_w=img_w, img_h=img_h,
        )
        annotations.append(ann)
    return annotations


def build_class_mapping(class_groups: dict[str, list[str]]) -> dict[str, str]:
    """Build a flat source->target mapping from class groups.

    Parameters
    ----------
    class_groups:
        ``{"vehicle": ["car", "truck", "bus"], "person": ["person"], ...}``

    Returns
    -------
    ``{"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "person": "person", ...}``
    """
    mapping: dict[str, str] = {}
    for group_name, sources in class_groups.items():
        for src in sources:
            mapping[src.lower()] = group_name
    return mapping


def auto_label(
    image_paths: list[Path],
    model_name: str,
    base_path: str,
    processor: str,
    target_classes: list[str],
    class_mapping: dict[str, str] | None = None,
    min_confidence: float = 0.3,
) -> dict[Path, list[Annotation]]:
    """Run a single YOLO model on images and return pre-annotations.

    Parameters
    ----------
    image_paths:
        Images to auto-label.
    model_name:
        Specific model to use (e.g. ``"yolo11s"``).
    base_path:
        Model base path for pyzm Detector.
    processor:
        ``"cpu"`` or ``"gpu"``.
    target_classes:
        Final ordered class names (group names if grouping is used).
    class_mapping:
        Optional source->target mapping from :func:`build_class_mapping`.
    min_confidence:
        Minimum detection confidence to keep (default 0.3, lower than
        production to catch more candidates for human review).
    """
    import cv2
    from pyzm.ml.detector import Detector

    det = Detector(
        models=[model_name],
        base_path=base_path,
        processor=processor,
    )

    results: dict[Path, list[Annotation]] = {}
    for img_path in image_paths:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image: %s", img_path)
            results[img_path] = []
            continue

        h, w = img.shape[:2]
        try:
            det_result = det.detect(img)
        except Exception:
            logger.exception("Detection failed for %s", img_path)
            results[img_path] = []
            continue

        results[img_path] = detections_to_annotations(
            det_result, target_classes,
            img_w=w, img_h=h,
            class_mapping=class_mapping,
            min_confidence=min_confidence,
        )
        logger.info(
            "Auto-labeled %s: %d annotations",
            img_path.name, len(results[img_path]),
        )

    return results
