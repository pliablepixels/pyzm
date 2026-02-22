"""Verification-first data model for the training UI.

Users upload images, auto-detect runs, and each detection is reviewed:
approve, delete, rename, reshape, or add missing objects. Classes emerge
naturally from user corrections rather than being defined upfront.

Persistence is via a simple ``verifications.json`` in the project directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pyzm.train.dataset import Annotation

logger = logging.getLogger("pyzm.train")


class DetectionStatus(str, Enum):
    """Review status of an individual detection."""

    PENDING = "pending"
    APPROVED = "approved"
    DELETED = "deleted"
    RENAMED = "renamed"
    RESHAPED = "reshaped"
    ADDED = "added"


@dataclass
class VerifiedDetection:
    """A single detection with its review status.

    Parameters
    ----------
    detection_id : str
        Unique ID within the image (e.g. ``"det_0"``).
    original : Annotation
        The annotation as produced by auto-detect (or user-drawn).
    status : DetectionStatus
        Current review status.
    new_label : str | None
        Overridden label (for RENAMED detections).
    adjusted : Annotation | None
        Adjusted bounding box (for RESHAPED detections).
    original_label : str
        The label from the model at detection time.
    """

    detection_id: str
    original: Annotation
    status: DetectionStatus = DetectionStatus.PENDING
    new_label: str | None = None
    adjusted: Annotation | None = None
    original_label: str = ""
    confidence: float | None = None

    @property
    def effective_label(self) -> str:
        """The label that should be used for training."""
        if self.status == DetectionStatus.RENAMED and self.new_label:
            return self.new_label
        return self.original_label

    @property
    def effective_annotation(self) -> Annotation:
        """The annotation that should be used for training."""
        if self.status == DetectionStatus.RESHAPED and self.adjusted:
            return self.adjusted
        return self.original

    @property
    def is_active(self) -> bool:
        """Whether this detection contributes to training data."""
        return self.status not in (DetectionStatus.PENDING, DetectionStatus.DELETED)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "detection_id": self.detection_id,
            "original": {
                "class_id": self.original.class_id,
                "cx": self.original.cx,
                "cy": self.original.cy,
                "w": self.original.w,
                "h": self.original.h,
            },
            "status": self.status.value,
            "original_label": self.original_label,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.new_label is not None:
            d["new_label"] = self.new_label
        if self.adjusted is not None:
            d["adjusted"] = {
                "class_id": self.adjusted.class_id,
                "cx": self.adjusted.cx,
                "cy": self.adjusted.cy,
                "w": self.adjusted.w,
                "h": self.adjusted.h,
            }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerifiedDetection:
        orig = d["original"]
        original = Annotation(
            class_id=orig["class_id"],
            cx=orig["cx"], cy=orig["cy"],
            w=orig["w"], h=orig["h"],
        )
        adjusted = None
        if "adjusted" in d:
            adj = d["adjusted"]
            adjusted = Annotation(
                class_id=adj["class_id"],
                cx=adj["cx"], cy=adj["cy"],
                w=adj["w"], h=adj["h"],
            )
        return cls(
            detection_id=d["detection_id"],
            original=original,
            status=DetectionStatus(d["status"]),
            new_label=d.get("new_label"),
            adjusted=adjusted,
            original_label=d.get("original_label", ""),
            confidence=d.get("confidence"),
        )


@dataclass
class ImageVerification:
    """Verification state for a single image."""

    image_name: str
    detections: list[VerifiedDetection] = field(default_factory=list)
    fully_reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "detections": [d.to_dict() for d in self.detections],
            "fully_reviewed": self.fully_reviewed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageVerification:
        return cls(
            image_name=d["image_name"],
            detections=[VerifiedDetection.from_dict(det) for det in d.get("detections", [])],
            fully_reviewed=d.get("fully_reviewed", False),
        )

    @property
    def active_detections(self) -> list[VerifiedDetection]:
        """Detections that contribute to training."""
        return [d for d in self.detections if d.is_active]

    @property
    def pending_count(self) -> int:
        return sum(1 for d in self.detections if d.status == DetectionStatus.PENDING)


class VerificationStore:
    """Persists verification state for all images in a project.

    Data is stored in ``<project_dir>/verifications.json``.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._path = self.project_dir / "verifications.json"
        self._data: dict[str, ImageVerification] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                for entry in raw.get("images", []):
                    iv = ImageVerification.from_dict(entry)
                    self._data[iv.image_name] = iv
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not load verifications.json, starting fresh")
                self._data = {}

    def save(self) -> None:
        payload = {
            "images": [iv.to_dict() for iv in self._data.values()],
        }
        self._path.write_text(json.dumps(payload, indent=2))

    def get(self, image_name: str) -> ImageVerification | None:
        return self._data.get(image_name)

    def set(self, iv: ImageVerification) -> None:
        self._data[iv.image_name] = iv

    def remove(self, image_name: str) -> None:
        """Remove an image from the store."""
        self._data.pop(image_name, None)

    def all_images(self) -> list[str]:
        return list(self._data.keys())

    def pending_count(self) -> int:
        """Total pending detections across all images."""
        return sum(iv.pending_count for iv in self._data.values())

    def reviewed_images_count(self) -> int:
        """Number of images fully reviewed."""
        return sum(1 for iv in self._data.values() if iv.fully_reviewed)

    def build_class_list(self) -> list[str]:
        """Derive unique class names from active (approved/renamed/added) detections.

        Returns sorted list of unique effective labels.
        """
        labels: set[str] = set()
        for iv in self._data.values():
            for det in iv.active_detections:
                label = det.effective_label
                if label:
                    labels.add(label)
        return sorted(labels)

    def per_class_image_counts(self, class_names: list[str] | None = None) -> dict[str, int]:
        """Count how many images contain at least one active detection per class.

        Parameters
        ----------
        class_names : list[str] | None
            If provided, count only these classes. Otherwise derive from data.
        """
        if class_names is None:
            class_names = self.build_class_list()

        counts: dict[str, int] = {name: 0 for name in class_names}
        for iv in self._data.values():
            seen: set[str] = set()
            for det in iv.active_detections:
                label = det.effective_label
                if label and label not in seen:
                    seen.add(label)
                    if label in counts:
                        counts[label] += 1
        return counts

    def corrected_classes(self) -> dict[str, dict[str, int]]:
        """Identify classes where the user corrected the model.

        Returns a dict mapping class names to correction reason counts.
        Only DELETED, RENAMED, RESHAPED, and ADDED statuses count as
        corrections; APPROVED and PENDING are not.

        Example::

            {"dog": {"added": 3, "renamed_to": 1}, "person": {"deleted": 1}}
        """
        result: dict[str, dict[str, int]] = {}

        def _bump(cls: str, reason: str) -> None:
            if cls not in result:
                result[cls] = {}
            result[cls][reason] = result[cls].get(reason, 0) + 1

        for iv in self._data.values():
            for det in iv.detections:
                if det.status == DetectionStatus.DELETED:
                    _bump(det.original_label, "deleted")
                elif det.status == DetectionStatus.RENAMED:
                    _bump(det.original_label, "renamed_from")
                    _bump(det.effective_label, "renamed_to")
                elif det.status == DetectionStatus.RESHAPED:
                    _bump(det.effective_label, "reshaped")
                elif det.status == DetectionStatus.ADDED:
                    _bump(det.effective_label, "added")

        return result

    def classes_needing_upload(
        self, min_images: int = 15,
    ) -> list[dict[str, Any]]:
        """Return corrected classes that need more training images.

        Only classes the user actually changed (added, renamed-to, reshaped)
        are included.  Old wrong labels (only renamed-from / deleted) are
        excluded — the user doesn't want to train on those.

        Returns a list of dicts sorted by total corrections (most first)::

            [{"class_name": str, "current_images": int, "target_images": int,
              "corrections": dict, "reason_summary": str}, ...]
        """
        corrections = self.corrected_classes()
        if not corrections:
            return []

        # Skip old wrong labels (only renamed_from and/or deleted).
        _NEGATIVE_ONLY = {"renamed_from", "deleted"}

        counts = self.per_class_image_counts(list(corrections.keys()))
        result: list[dict[str, Any]] = []
        for cls, reasons in corrections.items():
            if set(reasons.keys()) <= _NEGATIVE_ONLY:
                continue
            current = counts.get(cls, 0)
            if current >= min_images:
                continue
            total = sum(reasons.values())
            parts = [f"{reason} in {n}" for reason, n in reasons.items()]
            result.append({
                "class_name": cls,
                "current_images": current,
                "target_images": min_images,
                "corrections": reasons,
                "reason_summary": ", ".join(parts),
                "_total_corrections": total,
            })
        result.sort(key=lambda x: x["_total_corrections"], reverse=True)
        for entry in result:
            del entry["_total_corrections"]
        return result

    def has_modifications(self) -> bool:
        """Check whether any detection was modified during review.

        Returns ``True`` if any detection has status DELETED, RENAMED,
        RESHAPED, or ADDED — i.e. the user changed something beyond
        simple approval.
        """
        _MODIFIED = {
            DetectionStatus.DELETED,
            DetectionStatus.RENAMED,
            DetectionStatus.RESHAPED,
            DetectionStatus.ADDED,
        }
        return any(
            det.status in _MODIFIED
            for iv in self._data.values()
            for det in iv.detections
        )

    def finalized_annotations(
        self,
        image_name: str,
        class_name_to_id: dict[str, int],
    ) -> list[Annotation]:
        """Return final Annotation list for an image, ready for YOLO training.

        Maps effective labels to class IDs. Only includes active detections
        whose effective label exists in class_name_to_id.
        """
        iv = self._data.get(image_name)
        if iv is None:
            return []

        annotations: list[Annotation] = []
        for det in iv.active_detections:
            label = det.effective_label
            if label not in class_name_to_id:
                continue
            ann = det.effective_annotation
            annotations.append(Annotation(
                class_id=class_name_to_id[label],
                cx=ann.cx,
                cy=ann.cy,
                w=ann.w,
                h=ann.h,
            ))
        return annotations
