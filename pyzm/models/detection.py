"""Detection result models.

Results are proper objects instead of parallel arrays - no more index-mismatch
bugs from ``matched_data['labels'][i]`` / ``matched_data['boxes'][i]``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box (x1, y1) top-left to (x2, y2) bottom-right."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_polygon_coords(self) -> list[tuple[int, int]]:
        """Return four corners suitable for Shapely ``Polygon()``."""
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2),
        ]


# ---------------------------------------------------------------------------
# Detection (single object)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object/face/plate."""
    label: str
    confidence: float
    bbox: BBox
    model_name: str = ""
    detection_type: str = "object"

    def matches_pattern(self, pattern: str) -> bool:
        return bool(re.match(pattern, self.label))


# ---------------------------------------------------------------------------
# DetectionResult (aggregate)
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Aggregate result returned by :meth:`Detector.detect`."""
    detections: list[Detection] = field(default_factory=list)
    frame_id: int | str | None = None
    image: "np.ndarray | None" = field(default=None, repr=False)
    image_dimensions: dict[str, tuple[int, int] | None] = field(default_factory=dict)

    # Error boxes: detected but filtered out (wrong zone, wrong pattern, etc.)
    error_boxes: list[BBox] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return len(self.detections) > 0

    @property
    def labels(self) -> list[str]:
        return [d.label for d in self.detections]

    @property
    def confidences(self) -> list[float]:
        return [d.confidence for d in self.detections]

    @property
    def boxes(self) -> list[list[int]]:
        return [d.bbox.as_list() for d in self.detections]

    @property
    def summary(self) -> str:
        """Human-readable one-liner, e.g. ``person:97% car:85%``."""
        parts = [f"{d.label}:{d.confidence:.0%}" for d in self.detections]
        return " ".join(parts)

    def filter_by_pattern(self, pattern: str) -> "DetectionResult":
        """Return a new result keeping only detections whose label matches *pattern*."""
        kept = [d for d in self.detections if d.matches_pattern(pattern)]
        return DetectionResult(
            detections=kept,
            frame_id=self.frame_id,
            image=self.image,
            image_dimensions=self.image_dimensions,
        )

    def annotate(self, **draw_kwargs: object) -> "np.ndarray":
        """Draw bounding boxes on ``self.image`` and return the annotated copy.

        Extra *draw_kwargs* are forwarded to the internal bbox drawing function.
        If OpenCV is not available this raises ``ImportError``.
        """
        import cv2  # noqa: F811
        import numpy as np  # noqa: F811

        if self.image is None:
            raise ValueError("No image attached to this DetectionResult")

        image = self.image.copy()

        slate_colors = [
            (39, 174, 96),
            (142, 68, 173),
            (0, 129, 254),
            (254, 60, 113),
            (243, 134, 48),
            (91, 177, 47),
        ]

        for i, det in enumerate(self.detections):
            color = slate_colors[i % len(slate_colors)]
            b = det.bbox
            label_text = f"{det.label} {det.confidence:.0%}"
            cv2.rectangle(image, (b.x1, b.y1), (b.x2, b.y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label_text, font, 0.8, 1)[0]
            cv2.rectangle(
                image,
                (b.x1, b.y1 - text_size[1] - 4),
                (b.x1 + text_size[0] + 4, b.y1),
                color,
                -1,
            )
            cv2.putText(image, label_text, (b.x1 + 2, b.y1 - 2), font, 0.8, (255, 255, 255), 1)

        for eb in self.error_boxes:
            cv2.rectangle(image, (eb.x1, eb.y1), (eb.x2, eb.y2), (0, 0, 255), 1)

        return image

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionResult":
        """Reconstruct a DetectionResult from a wire dict (inverse of ``to_dict``).

        The *image* field is left as ``None`` — the caller already has the
        original image and can attach it after reconstruction.
        """
        labels = data.get("labels", [])
        boxes = data.get("boxes", [])
        confidences = data.get("confidences", [])
        model_names = data.get("model_names", [])

        detections: list[Detection] = []
        for i, label in enumerate(labels):
            bbox = BBox(*boxes[i]) if i < len(boxes) else BBox(0, 0, 0, 0)
            conf = confidences[i] if i < len(confidences) else 0.0
            mname = model_names[i] if i < len(model_names) else ""
            detections.append(Detection(label=label, confidence=conf, bbox=bbox, model_name=mname))

        error_boxes = [BBox(*eb) for eb in data.get("error_boxes", [])]

        return cls(
            detections=detections,
            frame_id=data.get("frame_id"),
            image=None,
            image_dimensions=data.get("image_dimensions", {}),
            error_boxes=error_boxes,
        )

    # -- Convenience for backward compat with dict access -----------------

    def to_dict(self) -> dict:
        """Return data in the ``matched_data`` dict format."""
        return {
            "boxes": self.boxes,
            "labels": self.labels,
            "confidences": self.confidences,
            "frame_id": self.frame_id,
            "image_dimensions": self.image_dimensions,
            "image": self.image,
            "error_boxes": [eb.as_list() for eb in self.error_boxes],
            "model_names": [d.model_name for d in self.detections],
            "polygons": [],
        }
