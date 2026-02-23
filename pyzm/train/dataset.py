"""YOLO-format dataset management for fine-tuning.

Handles image/label storage, train/val splitting, YAML generation,
and dataset quality checks.

Directory layout::

    <project_dir>/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    ├── dataset.yaml
    └── project.json
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pyzm.train")


@dataclass
class Annotation:
    """Single bounding-box annotation in YOLO normalised format."""

    class_id: int
    cx: float  # centre-x, normalised 0-1
    cy: float  # centre-y, normalised 0-1
    w: float   # width, normalised 0-1
    h: float   # height, normalised 0-1

    def __post_init__(self) -> None:
        self.cx = max(0.0, min(1.0, self.cx))
        self.cy = max(0.0, min(1.0, self.cy))
        self.w = max(0.0, min(1.0, self.w))
        self.h = max(0.0, min(1.0, self.h))

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> Annotation:
        """Parse a YOLO annotation line.

        Supports standard box format (class_id cx cy w h) and polygon/
        segmentation format (class_id x1 y1 x2 y2 ... xn yn) used by
        Roboflow exports.  Polygons are converted to bounding boxes.
        """
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        if len(coords) == 4:
            return cls(class_id=class_id, cx=coords[0], cy=coords[1],
                       w=coords[2], h=coords[3])
        if len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            return cls(class_id=class_id,
                       cx=(x_min + x_max) / 2, cy=(y_min + y_max) / 2,
                       w=x_max - x_min, h=y_max - y_min)
        raise ValueError(f"Expected 4 or 6+ even coordinates, got {len(coords)}")


@dataclass
class QualityWarning:
    """A single dataset quality issue."""

    level: str   # "warning" or "error"
    message: str


@dataclass
class QualityReport:
    """Dataset quality assessment."""

    total_images: int = 0
    annotated_images: int = 0
    unannotated_images: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    warnings: list[QualityWarning] = field(default_factory=list)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* to *dst*, falling back to copy if linking fails."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class YOLODataset:
    """Manage a YOLO-format dataset within a project directory.

    Parameters
    ----------
    project_dir : Path
        Root of the training project (e.g. ``~/.pyzm/training/my_project``).
    classes : list[str]
        Ordered class names.  Index in this list == class_id in labels.
    """

    def __init__(
        self,
        project_dir: Path,
        classes: list[str],
        class_groups: dict[str, list[str]] | None = None,
        settings: dict[str, object] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.classes = list(classes)
        self.class_groups: dict[str, list[str]] = class_groups or {}
        self.settings: dict[str, object] = settings or {}
        self._staged_cache: list[Path] | None = None

        # Unsplit staging area
        self._images_dir = self.project_dir / "images" / "all"
        self._labels_dir = self.project_dir / "labels" / "all"

        # Train/val dirs
        self._train_images = self.project_dir / "images" / "train"
        self._train_labels = self.project_dir / "labels" / "train"
        self._val_images = self.project_dir / "images" / "val"
        self._val_labels = self.project_dir / "labels" / "val"

    # ------------------------------------------------------------------
    # Project initialisation
    # ------------------------------------------------------------------

    def init_project(self) -> None:
        """Create directory structure and save project metadata."""
        for d in (
            self._images_dir, self._labels_dir,
            self._train_images, self._train_labels,
            self._val_images, self._val_labels,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self._save_project_json()

    def _save_project_json(self) -> None:
        meta = {
            "classes": self.classes,
            "class_groups": self.class_groups,
            "settings": self.settings,
        }
        (self.project_dir / "project.json").write_text(
            json.dumps(meta, indent=2)
        )

    def set_setting(self, key: str, value: object) -> None:
        """Update a single project setting and persist."""
        self.settings[key] = value
        self._save_project_json()

    def get_setting(self, key: str, default: object = None) -> object:
        """Read a project setting."""
        return self.settings.get(key, default)

    def set_classes(
        self,
        classes: list[str],
        class_groups: dict[str, list[str]] | None = None,
    ) -> None:
        """Replace the class list (and optionally class groups) and persist."""
        self.classes = list(classes)
        if class_groups is not None:
            self.class_groups = class_groups
        self._save_project_json()

    @classmethod
    def load(cls, project_dir: Path) -> YOLODataset:
        """Load an existing project from its ``project.json``."""
        meta = json.loads((Path(project_dir) / "project.json").read_text())
        return cls(
            project_dir=project_dir,
            classes=meta["classes"],
            class_groups=meta.get("class_groups", {}),
            settings=meta.get("settings", {}),
        )

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------

    def add_image(
        self,
        image_path: Path,
        annotations: list[Annotation],
    ) -> Path:
        """Copy *image_path* into the staging area and write its label file.

        Returns the destination image path.
        """
        image_path = Path(image_path)
        dest = self._images_dir / image_path.name

        # Avoid overwriting — append a counter if name collides
        if dest.exists() and not dest.samefile(image_path):
            stem = image_path.stem
            suffix = image_path.suffix
            counter = 1
            while dest.exists():
                dest = self._images_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if not dest.exists():
            shutil.copy2(image_path, dest)

        label_path = self._labels_dir / f"{dest.stem}.txt"
        label_path.write_text(
            "\n".join(a.to_yolo_line() for a in annotations) + "\n"
            if annotations
            else ""
        )
        self._staged_cache = None  # invalidate cache
        return dest

    def remove_image(self, image_name: str) -> None:
        """Delete an image and its label file from staging."""
        img = self._images_dir / image_name
        lbl = self._labels_dir / (Path(image_name).stem + ".txt")
        if img.exists():
            img.unlink()
        if lbl.exists():
            lbl.unlink()
        self._staged_cache = None  # invalidate cache

    def update_annotations(
        self,
        image_name: str,
        annotations: list[Annotation],
    ) -> None:
        """Overwrite the label file for an existing image in staging."""
        label_path = self._labels_dir / f"{Path(image_name).stem}.txt"
        label_path.write_text(
            "\n".join(a.to_yolo_line() for a in annotations) + "\n"
            if annotations
            else ""
        )

    # ------------------------------------------------------------------
    # Image listing
    # ------------------------------------------------------------------

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def staged_images(self) -> list[Path]:
        """Return all images in the staging (``all``) directory.

        Results are cached in memory; the cache is invalidated by
        :meth:`add_image`.
        """
        if self._staged_cache is not None:
            return self._staged_cache
        if not self._images_dir.exists():
            return []
        self._staged_cache = sorted(
            p for p in self._images_dir.iterdir()
            if p.suffix.lower() in self._IMG_EXTS
        )
        return self._staged_cache

    def annotations_for(self, image_name: str) -> list[Annotation]:
        """Load annotations for a staged image by filename."""
        label_path = self._labels_dir / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            return []
        lines = label_path.read_text().strip().splitlines()
        return [Annotation.from_yolo_line(ln) for ln in lines if ln.strip()]

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------

    def split(self, val_ratio: float = 0.2, seed: int = 42) -> None:
        """Split staged images into train/val sets.

        Clears any previous split, then hard-links (with copy fallback)
        from ``all/`` into ``train/`` and ``val/``.

        If a ``split_map`` setting exists (dict mapping image names to
        ``"train"`` or ``"val"``), those assignments are honored.
        Unassigned images are randomly split using *val_ratio*.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive)")

        # Clean previous split
        for d in (
            self._train_images, self._train_labels,
            self._val_images, self._val_labels,
        ):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        images = self.staged_images()
        if not images:
            logger.warning("No images to split")
            return

        split_map: dict[str, str] = self.settings.get("split_map", {}) or {}

        # Separate pre-assigned from unassigned
        val_set: set[Path] = set()
        train_set: set[Path] = set()
        unassigned: list[Path] = []

        for img in images:
            assignment = split_map.get(img.name)
            if assignment == "val":
                val_set.add(img)
            elif assignment == "train":
                train_set.add(img)
            else:
                unassigned.append(img)

        # Randomly split unassigned images
        if unassigned:
            rng = random.Random(seed)
            rng.shuffle(unassigned)
            n_val = max(0, int(len(unassigned) * val_ratio))
            for img in unassigned[:n_val]:
                val_set.add(img)
            for img in unassigned[n_val:]:
                train_set.add(img)

        # Ensure at least 1 val image
        if not val_set and train_set:
            moved = next(iter(train_set))
            train_set.discard(moved)
            val_set.add(moved)

        for img in images:
            label_src = self._labels_dir / f"{img.stem}.txt"
            if img in val_set:
                _link_or_copy(img, self._val_images / img.name)
                if label_src.exists():
                    _link_or_copy(label_src, self._val_labels / f"{img.stem}.txt")
            else:
                _link_or_copy(img, self._train_images / img.name)
                if label_src.exists():
                    _link_or_copy(label_src, self._train_labels / f"{img.stem}.txt")

        logger.info(
            "Split: %d train, %d val (ratio=%.2f)",
            len(train_set), len(val_set), val_ratio,
        )

    # ------------------------------------------------------------------
    # dataset.yaml generation
    # ------------------------------------------------------------------

    def generate_yaml(self) -> Path:
        """Write ``dataset.yaml`` for Ultralytics and return its path."""
        yaml_path = self.project_dir / "dataset.yaml"

        names_block = "\n".join(
            f"  {i}: {name}" for i, name in enumerate(self.classes)
        )
        content = (
            f"path: {self.project_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"\n"
            f"nc: {len(self.classes)}\n"
            f"names:\n{names_block}\n"
        )
        yaml_path.write_text(content)
        return yaml_path

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------

    def quality_report(self) -> QualityReport:
        """Analyse dataset and return quality warnings."""
        images = self.staged_images()
        per_class: dict[str, int] = {cls: 0 for cls in self.classes}
        annotated = 0
        unannotated = 0

        for img in images:
            anns = self.annotations_for(img.name)
            if anns:
                annotated += 1
                for a in anns:
                    if 0 <= a.class_id < len(self.classes):
                        per_class[self.classes[a.class_id]] += 1
            else:
                unannotated += 1

        report = QualityReport(
            total_images=len(images),
            annotated_images=annotated,
            unannotated_images=unannotated,
            per_class=per_class,
        )

        # Generate warnings
        if len(images) < 20:
            report.warnings.append(QualityWarning(
                level="warning",
                message="Very small dataset. Consider adding more images.",
            ))

        if unannotated > 0:
            report.warnings.append(QualityWarning(
                level="warning",
                message=f"{unannotated} images have no annotations — will be skipped.",
            ))

        counts = [c for c in per_class.values() if c > 0]
        if counts:
            max_count = max(per_class.values())
            for cls, count in per_class.items():
                if count == 0:
                    continue
                if count < 10:
                    report.warnings.append(QualityWarning(
                        level="warning",
                        message=(
                            f"Very few images for '{cls}'. "
                            "Model will likely not detect this reliably."
                        ),
                    ))
                elif count < 30:
                    report.warnings.append(QualityWarning(
                        level="warning",
                        message=(
                            f"Limited images for '{cls}'. "
                            "50+ recommended for good accuracy."
                        ),
                    ))

            # Imbalance check — collect all underrepresented classes
            dominant = max(per_class, key=per_class.get)  # type: ignore[arg-type]
            underrep = [
                cls for cls, count in per_class.items()
                if count > 0 and max_count / count > 3
            ]
            if underrep:
                worst = min(underrep, key=lambda c: per_class[c])
                report.warnings.append(QualityWarning(
                    level="warning",
                    message=(
                        f"Class imbalance: '{dominant}' dominates; "
                        f"most underrepresented is '{worst}' "
                        f"({max_count // per_class[worst]}x fewer)."
                    ),
                ))

        return report
