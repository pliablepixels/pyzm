"""Tests for pyzm.train.dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyzm.train.dataset import Annotation, QualityReport, YOLODataset


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_to_yolo_line(self):
        a = Annotation(class_id=2, cx=0.5, cy=0.3, w=0.2, h=0.4)
        line = a.to_yolo_line()
        assert line.startswith("2 ")
        parts = line.split()
        assert len(parts) == 5
        assert float(parts[1]) == pytest.approx(0.5)
        assert float(parts[2]) == pytest.approx(0.3)

    def test_from_yolo_line(self):
        a = Annotation.from_yolo_line("3 0.512345 0.678901 0.123456 0.234567")
        assert a.class_id == 3
        assert a.cx == pytest.approx(0.512345)
        assert a.cy == pytest.approx(0.678901)

    def test_clamps_out_of_range_high(self):
        a = Annotation(class_id=0, cx=1.2, cy=1.5, w=2.0, h=1.1)
        assert a.cx == 1.0
        assert a.cy == 1.0
        assert a.w == 1.0
        assert a.h == 1.0

    def test_clamps_out_of_range_low(self):
        a = Annotation(class_id=0, cx=-0.1, cy=-0.5, w=-0.3, h=-1.0)
        assert a.cx == 0.0
        assert a.cy == 0.0
        assert a.w == 0.0
        assert a.h == 0.0

    def test_clamp_preserves_valid_values(self):
        a = Annotation(class_id=1, cx=0.5, cy=0.3, w=0.2, h=0.4)
        assert a.cx == pytest.approx(0.5)
        assert a.cy == pytest.approx(0.3)
        assert a.w == pytest.approx(0.2)
        assert a.h == pytest.approx(0.4)

    def test_roundtrip(self):
        orig = Annotation(class_id=1, cx=0.55, cy=0.45, w=0.3, h=0.6)
        line = orig.to_yolo_line()
        back = Annotation.from_yolo_line(line)
        assert back.class_id == orig.class_id
        assert back.cx == pytest.approx(orig.cx, abs=1e-5)
        assert back.cy == pytest.approx(orig.cy, abs=1e-5)
        assert back.w == pytest.approx(orig.w, abs=1e-5)
        assert back.h == pytest.approx(orig.h, abs=1e-5)


# ---------------------------------------------------------------------------
# YOLODataset
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset(tmp_path: Path) -> YOLODataset:
    ds = YOLODataset(project_dir=tmp_path / "project", classes=["person", "car", "package"])
    ds.init_project()
    return ds


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Create 10 fake image files."""
    imgs_dir = tmp_path / "source_images"
    imgs_dir.mkdir()
    paths = []
    for i in range(10):
        p = imgs_dir / f"img_{i:03d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG-ish
        paths.append(p)
    return paths


class TestYOLODataset:
    def test_init_project_creates_dirs(self, dataset: YOLODataset):
        assert (dataset.project_dir / "images" / "all").is_dir()
        assert (dataset.project_dir / "labels" / "all").is_dir()
        assert (dataset.project_dir / "images" / "train").is_dir()
        assert (dataset.project_dir / "images" / "val").is_dir()
        assert (dataset.project_dir / "project.json").exists()

    def test_load_project(self, dataset: YOLODataset):
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.classes == ["person", "car", "package"]

    def test_class_groups_persist(self, tmp_path: Path):
        groups = {"vehicle": ["car", "truck"], "person": ["person"]}
        ds = YOLODataset(
            project_dir=tmp_path / "grouped",
            classes=["person", "vehicle"],
            class_groups=groups,
        )
        ds.init_project()
        loaded = YOLODataset.load(ds.project_dir)
        assert loaded.class_groups == groups
        assert loaded.classes == ["person", "vehicle"]

    def test_add_image(self, dataset: YOLODataset, sample_images: list[Path]):
        anns = [Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.4)]
        dest = dataset.add_image(sample_images[0], anns)
        assert dest.exists()
        assert (dataset._labels_dir / f"{dest.stem}.txt").exists()

    def test_add_image_dedup(self, dataset: YOLODataset, sample_images: list[Path]):
        """Adding an image with the same name gets a counter suffix."""
        dataset.add_image(sample_images[0], [])
        # Create a different file with same name in a different dir
        other_dir = sample_images[0].parent.parent / "other"
        other_dir.mkdir()
        dup = other_dir / sample_images[0].name
        dup.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        dest = dataset.add_image(dup, [])
        assert "_1" in dest.stem

    def test_add_image_no_annotations(self, dataset: YOLODataset, sample_images: list[Path]):
        dest = dataset.add_image(sample_images[0], [])
        label = dataset._labels_dir / f"{dest.stem}.txt"
        assert label.read_text() == ""

    def test_staged_images(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:3]:
            dataset.add_image(img, [])
        assert len(dataset.staged_images()) == 3

    def test_annotations_for(self, dataset: YOLODataset, sample_images: list[Path]):
        anns = [
            Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.4),
            Annotation(class_id=1, cx=0.2, cy=0.3, w=0.1, h=0.2),
        ]
        dest = dataset.add_image(sample_images[0], anns)
        loaded = dataset.annotations_for(dest.name)
        assert len(loaded) == 2
        assert loaded[0].class_id == 0
        assert loaded[1].class_id == 1

    def test_update_annotations(self, dataset: YOLODataset, sample_images: list[Path]):
        dest = dataset.add_image(sample_images[0], [])
        assert dataset.annotations_for(dest.name) == []

        new_anns = [Annotation(class_id=2, cx=0.5, cy=0.5, w=0.2, h=0.3)]
        dataset.update_annotations(dest.name, new_anns)
        loaded = dataset.annotations_for(dest.name)
        assert len(loaded) == 1
        assert loaded[0].class_id == 2

    def test_split_basic(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])
        dataset.split(val_ratio=0.2)

        train_imgs = list(dataset._train_images.iterdir())
        val_imgs = list(dataset._val_images.iterdir())
        assert len(train_imgs) + len(val_imgs) == 10
        assert len(val_imgs) == 2  # 20% of 10

    def test_split_copies_labels(self, dataset: YOLODataset, sample_images: list[Path]):
        anns = [Annotation(class_id=1, cx=0.5, cy=0.5, w=0.3, h=0.4)]
        for img in sample_images[:5]:
            dataset.add_image(img, anns)
        dataset.split(val_ratio=0.2)

        train_labels = list(dataset._train_labels.iterdir())
        val_labels = list(dataset._val_labels.iterdir())
        assert len(train_labels) + len(val_labels) == 5

    def test_split_invalid_ratio(self, dataset: YOLODataset):
        with pytest.raises(ValueError, match="val_ratio"):
            dataset.split(val_ratio=0.0)
        with pytest.raises(ValueError, match="val_ratio"):
            dataset.split(val_ratio=1.0)

    def test_split_deterministic(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        dataset.split(val_ratio=0.3, seed=42)
        val1 = sorted(p.name for p in dataset._val_images.iterdir())

        dataset.split(val_ratio=0.3, seed=42)
        val2 = sorted(p.name for p in dataset._val_images.iterdir())

        assert val1 == val2

    def test_split_honors_split_map(self, dataset: YOLODataset, sample_images: list[Path]):
        """Images pre-assigned via split_map go to their designated dirs."""
        for img in sample_images:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        staged = dataset.staged_images()
        # Assign first 2 images to val, next 3 to train
        split_map = {}
        for img in staged[:2]:
            split_map[img.name] = "val"
        for img in staged[2:5]:
            split_map[img.name] = "train"
        # Remaining 5 are unassigned (random split)
        dataset.set_setting("split_map", split_map)

        dataset.split(val_ratio=0.2)

        val_names = {p.name for p in dataset._val_images.iterdir()}
        train_names = {p.name for p in dataset._train_images.iterdir()}

        # Pre-assigned images must be in their designated dirs
        for img in staged[:2]:
            assert img.name in val_names, f"{img.name} should be in val"
        for img in staged[2:5]:
            assert img.name in train_names, f"{img.name} should be in train"

        # Total should be 10
        assert len(val_names) + len(train_names) == 10

    def test_split_mixed_assigned_and_random(self, dataset: YOLODataset, sample_images: list[Path]):
        """Unassigned images get randomly split while assigned ones are honored."""
        for img in sample_images[:4]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        staged = dataset.staged_images()
        # Assign only the first image to train
        dataset.set_setting("split_map", {staged[0].name: "train"})

        dataset.split(val_ratio=0.5)

        train_names = {p.name for p in dataset._train_images.iterdir()}
        val_names = {p.name for p in dataset._val_images.iterdir()}

        assert staged[0].name in train_names
        assert len(train_names) + len(val_names) == 4
        assert len(val_names) >= 1  # at least 1 val image guaranteed

    def test_split_hardlinks(self, dataset: YOLODataset, sample_images: list[Path]):
        """Verify split uses hardlinks (same inode) on supported systems."""
        import os

        for img in sample_images[:3]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])
        dataset.split(val_ratio=0.3)

        # Check that at least one train image shares inode with staging
        staged = dataset.staged_images()
        for staged_img in staged:
            train_path = dataset._train_images / staged_img.name
            val_path = dataset._val_images / staged_img.name
            dest = train_path if train_path.exists() else val_path
            if dest.exists():
                assert os.path.samefile(staged_img, dest), (
                    f"Expected hardlink (same inode) for {staged_img.name}"
                )
                break
        else:
            pytest.fail("No split image found to verify hardlink")

    def test_generate_yaml(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:3]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])
        dataset.split()

        yaml_path = dataset.generate_yaml()
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "nc: 3" in content
        assert "person" in content
        assert "car" in content
        assert "package" in content
        assert "train: images/train" in content
        assert "val: images/val" in content


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

class TestSetClasses:
    def test_set_classes_updates_and_persists(self, dataset: YOLODataset):
        dataset.set_classes(["cat", "dog"])
        assert dataset.classes == ["cat", "dog"]
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.classes == ["cat", "dog"]

    def test_set_classes_with_groups(self, dataset: YOLODataset):
        groups = {"animal": ["cat", "dog"]}
        dataset.set_classes(["animal"], class_groups=groups)
        assert dataset.class_groups == groups
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.class_groups == groups

    def test_set_classes_preserves_groups_when_none(self, tmp_path: Path):
        ds = YOLODataset(
            project_dir=tmp_path / "proj",
            classes=["a", "b"],
            class_groups={"a": ["x"], "b": ["y"]},
        )
        ds.init_project()
        ds.set_classes(["c", "d"])
        assert ds.class_groups == {"a": ["x"], "b": ["y"]}

    def test_set_classes_empty(self, dataset: YOLODataset):
        dataset.set_classes([])
        assert dataset.classes == []
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.classes == []


class TestSettings:
    def test_set_and_get_setting(self, dataset: YOLODataset):
        assert dataset.get_setting("auto_detect") is None
        assert dataset.get_setting("auto_detect", True) is True

        dataset.set_setting("auto_detect", False)
        assert dataset.get_setting("auto_detect") is False

        # Persists across reload
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.get_setting("auto_detect") is False

    def test_settings_default_empty(self, dataset: YOLODataset):
        assert dataset.settings == {}

    def test_settings_preserved_on_set_classes(self, dataset: YOLODataset):
        dataset.set_setting("auto_detect", True)
        dataset.set_classes(["cat", "dog"])
        loaded = YOLODataset.load(dataset.project_dir)
        assert loaded.get_setting("auto_detect") is True
        assert loaded.classes == ["cat", "dog"]


class TestQualityReport:
    def test_small_dataset_warning(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:5]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        report = dataset.quality_report()
        assert report.total_images == 5
        assert report.annotated_images == 5
        msgs = [w.message for w in report.warnings]
        assert any("Very small dataset" in m for m in msgs)

    def test_unannotated_warning(self, dataset: YOLODataset, sample_images: list[Path]):
        dataset.add_image(sample_images[0], [])
        dataset.add_image(sample_images[1], [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        report = dataset.quality_report()
        assert report.unannotated_images == 1
        msgs = [w.message for w in report.warnings]
        assert any("no annotations" in m for m in msgs)

    def test_few_images_warning(self, dataset: YOLODataset, sample_images: list[Path]):
        for img in sample_images[:3]:
            dataset.add_image(img, [Annotation(0, 0.5, 0.5, 0.3, 0.3)])

        report = dataset.quality_report()
        msgs = [w.message for w in report.warnings]
        assert any("Very few images for 'person'" in m for m in msgs)

    def test_per_class_counts(self, dataset: YOLODataset, sample_images: list[Path]):
        dataset.add_image(sample_images[0], [
            Annotation(0, 0.5, 0.5, 0.3, 0.3),
            Annotation(0, 0.2, 0.2, 0.1, 0.1),
        ])
        dataset.add_image(sample_images[1], [
            Annotation(1, 0.5, 0.5, 0.3, 0.3),
        ])

        report = dataset.quality_report()
        assert report.per_class["person"] == 2
        assert report.per_class["car"] == 1
        assert report.per_class["package"] == 0

    def test_imbalance_single_aggregate_warning(self, tmp_path: Path, sample_images: list[Path]):
        """Imbalance should produce exactly one aggregate warning, not one per class."""
        ds = YOLODataset(
            project_dir=tmp_path / "imbalance_proj",
            classes=["person", "car", "package"],
        )
        ds.init_project()
        # person: 40 annotations, car: 5, package: 3  => imbalanced
        for i, img in enumerate(sample_images):
            anns = []
            # 4 person annotations per image = 40 total
            for _ in range(4):
                anns.append(Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.3))
            if i < 5:
                anns.append(Annotation(class_id=1, cx=0.2, cy=0.2, w=0.1, h=0.1))
            if i < 3:
                anns.append(Annotation(class_id=2, cx=0.8, cy=0.8, w=0.1, h=0.1))
            ds.add_image(img, anns)

        report = ds.quality_report()
        imbalance_warnings = [
            w for w in report.warnings if "imbalance" in w.message.lower()
        ]
        assert len(imbalance_warnings) == 1, (
            f"Expected 1 aggregate imbalance warning, got {len(imbalance_warnings)}: "
            f"{[w.message for w in imbalance_warnings]}"
        )
        msg = imbalance_warnings[0].message
        assert "person" in msg  # dominant
        assert "package" in msg  # most underrepresented

    def test_empty_dataset(self, dataset: YOLODataset):
        report = dataset.quality_report()
        assert report.total_images == 0
        assert report.annotated_images == 0
        # Still warns about small dataset (0 < 20)
        msgs = [w.message for w in report.warnings]
        assert any("Very small dataset" in m for m in msgs)
