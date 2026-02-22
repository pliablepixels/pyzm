"""Tests for pyzm.train.local_import."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.local_import import (
    _find_images,
    _find_matching_label,
    _import_local_dataset,
    _import_raw_images,
    _infer_split,
    import_local_dataset,
    import_raw_images,
    validate_yolo_folder,
)
from pyzm.train.verification import (
    DetectionStatus,
    VerificationStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_folder(tmp_path: Path, *, names: dict | list | None = None,
                      flat: bool = True) -> Path:
    """Create a minimal YOLO dataset folder for testing."""
    folder = tmp_path / "yolo_ds"
    folder.mkdir()

    if names is None:
        names = {0: "person", 1: "car"}
    yaml_content = f"names: {names}\n"
    (folder / "data.yaml").write_text(yaml_content)

    if flat:
        img_dir = folder / "images"
        lbl_dir = folder / "labels"
    else:
        img_dir = folder / "images" / "train"
        lbl_dir = folder / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create dummy images (1x1 JPEG)
    from PIL import Image
    for i in range(3):
        img = Image.new("RGB", (100, 100), "red")
        img.save(str(img_dir / f"img{i:03d}.jpg"))
        (lbl_dir / f"img{i:03d}.txt").write_text(
            f"0 0.5 0.5 0.2 0.3\n1 0.7 0.7 0.1 0.1\n"
        )

    return folder


def _make_roboflow_folder(tmp_path: Path, *, polygon: bool = False) -> Path:
    """Create a Roboflow-style split YOLO dataset folder.

    Layout::

        folder/
          data.yaml
          train/images/  train/labels/
          valid/images/  valid/labels/
    """
    folder = tmp_path / "roboflow_ds"
    folder.mkdir()

    names = {0: "license_plate", 1: "vehicle"}
    (folder / "data.yaml").write_text(f"names: {names}\n")

    from PIL import Image

    for split, count in [("train", 3), ("valid", 2)]:
        img_dir = folder / split / "images"
        lbl_dir = folder / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(count):
            img = Image.new("RGB", (100, 100), "blue")
            img.save(str(img_dir / f"{split}_{i:03d}.jpg"))
            if polygon:
                # Polygon with 4 vertices (8 coords) forming a rectangle
                lbl_dir.joinpath(f"{split}_{i:03d}.txt").write_text(
                    "0 0.2 0.3 0.8 0.3 0.8 0.7 0.2 0.7\n"
                )
            else:
                lbl_dir.joinpath(f"{split}_{i:03d}.txt").write_text(
                    "0 0.5 0.5 0.6 0.4\n"
                )

    return folder


# ---------------------------------------------------------------------------
# validate_yolo_folder
# ---------------------------------------------------------------------------

class TestValidateYoloFolder:
    def test_valid_folder(self, tmp_path):
        folder = _make_yolo_folder(tmp_path)
        result = validate_yolo_folder(folder)
        assert isinstance(result, dict)
        assert "names" in result
        assert result["names"] == {0: "person", 1: "car"}
        assert "_splits" in result
        assert len(result["_splits"]) == 1

    def test_not_a_directory(self, tmp_path):
        result = validate_yolo_folder(tmp_path / "nonexistent")
        assert isinstance(result, str)
        assert "Not a directory" in result

    def test_missing_data_yaml(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        result = validate_yolo_folder(folder)
        assert isinstance(result, str)
        assert "Missing data.yaml" in result

    def test_missing_names_key(self, tmp_path):
        folder = tmp_path / "no_names"
        folder.mkdir()
        (folder / "data.yaml").write_text("nc: 2\n")
        (folder / "images").mkdir()
        (folder / "labels").mkdir()
        result = validate_yolo_folder(folder)
        assert isinstance(result, str)
        assert "names" in result

    def test_no_images_or_splits(self, tmp_path):
        """Neither standard images/labels nor split dirs exist."""
        folder = tmp_path / "no_layout"
        folder.mkdir()
        (folder / "data.yaml").write_text("names: {0: cat}\n")
        result = validate_yolo_folder(folder)
        assert isinstance(result, str)
        assert "No images/labels directories found" in result

    def test_names_as_list(self, tmp_path):
        folder = tmp_path / "list_names"
        folder.mkdir()
        (folder / "data.yaml").write_text("names:\n  - person\n  - car\n")
        (folder / "images").mkdir()
        (folder / "labels").mkdir()
        result = validate_yolo_folder(folder)
        assert isinstance(result, dict)
        assert result["names"] == {0: "person", 1: "car"}

    def test_invalid_yaml(self, tmp_path):
        folder = tmp_path / "bad_yaml"
        folder.mkdir()
        (folder / "data.yaml").write_text("{{invalid yaml")
        result = validate_yolo_folder(folder)
        assert isinstance(result, str)
        assert "parse" in result.lower() or "Failed" in result

    def test_roboflow_layout(self, tmp_path):
        """Roboflow-style split dirs are discovered."""
        folder = _make_roboflow_folder(tmp_path)
        result = validate_yolo_folder(folder)
        assert isinstance(result, dict)
        splits = result["_splits"]
        assert len(splits) == 2  # train + valid
        # Each split is (images_dir, labels_dir)
        split_names = sorted(s[0].parent.name for s in splits)
        assert split_names == ["train", "valid"]


# ---------------------------------------------------------------------------
# _find_images
# ---------------------------------------------------------------------------

class TestFindImages:
    def test_flat_layout(self, tmp_path):
        folder = _make_yolo_folder(tmp_path, flat=True)
        images = _find_images(folder / "images")
        assert len(images) == 3
        assert all(p.suffix == ".jpg" for p in images)

    def test_nested_layout(self, tmp_path):
        folder = _make_yolo_folder(tmp_path, flat=False)
        images = _find_images(folder / "images")
        assert len(images) == 3

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert _find_images(d) == []


# ---------------------------------------------------------------------------
# _find_matching_label
# ---------------------------------------------------------------------------

class TestFindMatchingLabel:
    def test_flat_match(self, tmp_path):
        folder = _make_yolo_folder(tmp_path, flat=True)
        images = _find_images(folder / "images")
        label = _find_matching_label(images[0], folder / "images", folder / "labels")
        assert label is not None
        assert label.suffix == ".txt"

    def test_nested_match(self, tmp_path):
        folder = _make_yolo_folder(tmp_path, flat=False)
        images = _find_images(folder / "images")
        label = _find_matching_label(images[0], folder / "images", folder / "labels")
        assert label is not None

    def test_no_match(self, tmp_path):
        folder = _make_yolo_folder(tmp_path, flat=True)
        # Create image without label
        from PIL import Image
        img = Image.new("RGB", (10, 10))
        img.save(str(folder / "images" / "orphan.jpg"))
        orphan = folder / "images" / "orphan.jpg"
        label = _find_matching_label(orphan, folder / "images", folder / "labels")
        assert label is None


# ---------------------------------------------------------------------------
# Annotation.from_yolo_line
# ---------------------------------------------------------------------------

class TestAnnotationFromYoloLine:
    def test_standard_box(self):
        ann = Annotation.from_yolo_line("2 0.5 0.5 0.2 0.3")
        assert ann.class_id == 2
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.5)
        assert ann.w == pytest.approx(0.2)
        assert ann.h == pytest.approx(0.3)

    def test_polygon_rectangle(self):
        """4-vertex polygon (8 coords) → bounding box."""
        # Rectangle: (0.2,0.3) (0.8,0.3) (0.8,0.7) (0.2,0.7)
        ann = Annotation.from_yolo_line("0 0.2 0.3 0.8 0.3 0.8 0.7 0.2 0.7")
        assert ann.class_id == 0
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.5)
        assert ann.w == pytest.approx(0.6)
        assert ann.h == pytest.approx(0.4)

    def test_polygon_triangle(self):
        """3-vertex polygon (6 coords) → bounding box."""
        # Triangle: (0.1,0.2) (0.9,0.2) (0.5,0.8)
        ann = Annotation.from_yolo_line("1 0.1 0.2 0.9 0.2 0.5 0.8")
        assert ann.class_id == 1
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.5)
        assert ann.w == pytest.approx(0.8)
        assert ann.h == pytest.approx(0.6)

    def test_polygon_pentagon(self):
        """5-vertex polygon (10 coords) → bounding box."""
        ann = Annotation.from_yolo_line(
            "0 0.1 0.2 0.9 0.2 0.9 0.8 0.5 0.9 0.1 0.8"
        )
        assert ann.class_id == 0
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.55)
        assert ann.w == pytest.approx(0.8)
        assert ann.h == pytest.approx(0.7)

    def test_odd_coords_raises(self):
        """Odd number of coordinates (not 4 and not even 6+) raises."""
        with pytest.raises(ValueError, match="Expected 4 or 6\\+ even"):
            Annotation.from_yolo_line("0 0.1 0.2 0.3 0.4 0.5")

    def test_too_few_coords_raises(self):
        with pytest.raises(ValueError, match="Expected 4 or 6\\+ even"):
            Annotation.from_yolo_line("0 0.1 0.2")


# ---------------------------------------------------------------------------
# _import_local_dataset (integration)
# ---------------------------------------------------------------------------

class TestImportLocalDataset:
    @pytest.fixture
    def workspace(self, tmp_path):
        pdir = tmp_path / "workspace"
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        store = VerificationStore(pdir)
        return ds, store

    def test_import_flat(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        # Mock st.progress since we're not in a Streamlit context
        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count, det_count = _import_local_dataset(ds, store, splits, names_map)

        assert img_count == 3
        assert det_count == 6  # 2 detections per image * 3 images

        # All images should be in staging
        staged = ds.staged_images()
        assert len(staged) == 3

        # All should be fully reviewed with APPROVED detections
        for img in staged:
            iv = store.get(img.name)
            assert iv is not None
            assert iv.fully_reviewed is True
            assert len(iv.detections) == 2
            for det in iv.detections:
                assert det.status == DetectionStatus.APPROVED
                assert det.original_label in ("person", "car")

    def test_import_nested(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=False)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images" / "train", folder / "labels" / "train")]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count, det_count = _import_local_dataset(ds, store, splits, names_map)

        assert img_count == 3
        assert det_count == 6

    def test_import_images_without_labels(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        # Remove one label file
        (folder / "labels" / "img000.txt").unlink()
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count, det_count = _import_local_dataset(ds, store, splits, names_map)

        assert img_count == 3
        assert det_count == 4  # Only 2 images had labels

        # The unlabelled image should still be fully_reviewed with no detections
        staged = ds.staged_images()
        for img in staged:
            iv = store.get(img.name)
            assert iv is not None
            assert iv.fully_reviewed is True

    def test_import_empty_folder(self, tmp_path, workspace):
        ds, store = workspace
        folder = tmp_path / "empty_ds"
        folder.mkdir()
        (folder / "images").mkdir()
        (folder / "labels").mkdir()
        splits = [(folder / "images", folder / "labels")]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count, det_count = _import_local_dataset(ds, store, splits, {})

        assert img_count == 0
        assert det_count == 0

    def test_import_roboflow_splits(self, tmp_path, workspace):
        """Import from Roboflow-style train/valid split dirs."""
        ds, store = workspace
        folder = _make_roboflow_folder(tmp_path)
        names_map = {0: "license_plate", 1: "vehicle"}

        result = validate_yolo_folder(folder)
        assert isinstance(result, dict)
        splits = result["_splits"]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count, det_count = _import_local_dataset(ds, store, splits, names_map)

        assert img_count == 5  # 3 train + 2 valid
        assert det_count == 5  # 1 detection per image

    def test_import_polygon_labels(self, tmp_path, workspace):
        """Polygon annotations are converted to bounding boxes on import."""
        ds, store = workspace
        folder = _make_roboflow_folder(tmp_path, polygon=True)
        names_map = {0: "license_plate", 1: "vehicle"}

        result = validate_yolo_folder(folder)
        assert isinstance(result, dict)
        splits = result["_splits"]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count, det_count = _import_local_dataset(ds, store, splits, names_map)

        assert img_count == 5
        assert det_count == 5

        # Verify polygon was converted to bbox correctly
        staged = ds.staged_images()
        iv = store.get(staged[0].name)
        assert iv is not None
        ann = iv.detections[0].original
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.5)
        assert ann.w == pytest.approx(0.6)
        assert ann.h == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _infer_split
# ---------------------------------------------------------------------------

class TestInferSplit:
    def test_roboflow_train(self, tmp_path):
        """Roboflow layout: train/images/img.jpg → 'train'."""
        images_dir = tmp_path / "train" / "images"
        images_dir.mkdir(parents=True)
        img = images_dir / "photo.jpg"
        img.touch()
        assert _infer_split(img, images_dir) == "train"

    def test_roboflow_valid(self, tmp_path):
        """Roboflow layout: valid/images/img.jpg → 'val'."""
        images_dir = tmp_path / "valid" / "images"
        images_dir.mkdir(parents=True)
        img = images_dir / "photo.jpg"
        img.touch()
        assert _infer_split(img, images_dir) == "val"

    def test_roboflow_val(self, tmp_path):
        """Roboflow layout: val/images/img.jpg → 'val'."""
        images_dir = tmp_path / "val" / "images"
        images_dir.mkdir(parents=True)
        img = images_dir / "photo.jpg"
        img.touch()
        assert _infer_split(img, images_dir) == "val"

    def test_standard_subdirs_train(self, tmp_path):
        """Standard layout: images/train/img.jpg → 'train'."""
        images_dir = tmp_path / "images"
        (images_dir / "train").mkdir(parents=True)
        img = images_dir / "train" / "photo.jpg"
        img.touch()
        assert _infer_split(img, images_dir) == "train"

    def test_standard_subdirs_val(self, tmp_path):
        """Standard layout: images/val/img.jpg → 'val'."""
        images_dir = tmp_path / "images"
        (images_dir / "val").mkdir(parents=True)
        img = images_dir / "val" / "photo.jpg"
        img.touch()
        assert _infer_split(img, images_dir) == "val"

    def test_flat_layout(self, tmp_path):
        """Flat layout: images/img.jpg → None."""
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)
        img = images_dir / "photo.jpg"
        img.touch()
        assert _infer_split(img, images_dir) is None


class TestImportWritesLabels:
    """Verify that labels are populated on import (not empty)."""

    @pytest.fixture
    def workspace(self, tmp_path):
        pdir = tmp_path / "workspace"
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        store = VerificationStore(pdir)
        return ds, store

    def test_labels_non_empty_after_import(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock
            _import_local_dataset(ds, store, splits, names_map)

        # Label files should be non-empty (annotations written at import)
        for img in ds.staged_images():
            anns = ds.annotations_for(img.name)
            assert len(anns) > 0, f"Label for {img.name} should not be empty"

    def test_import_records_split_map(self, tmp_path, workspace):
        """Roboflow import should populate split_map in settings."""
        ds, store = workspace
        folder = _make_roboflow_folder(tmp_path)
        names_map = {0: "license_plate", 1: "vehicle"}
        result = validate_yolo_folder(folder)
        assert isinstance(result, dict)
        splits = result["_splits"]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock
            _import_local_dataset(ds, store, splits, names_map)

        split_map = ds.get_setting("split_map")
        assert split_map is not None
        assert isinstance(split_map, dict)
        assert len(split_map) > 0
        # All values should be "train" or "val"
        for v in split_map.values():
            assert v in ("train", "val")

    def test_import_records_import_classes(self, tmp_path, workspace):
        """Import should save import_classes in settings."""
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock
            _import_local_dataset(ds, store, splits, names_map)

        import_classes = ds.get_setting("import_classes")
        assert import_classes == ["car", "person"]  # sorted

    def test_flat_import_no_split_map(self, tmp_path, workspace):
        """Flat layout import should not set split_map (no split info)."""
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        import unittest.mock as mock
        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock
            _import_local_dataset(ds, store, splits, names_map)

        split_map = ds.get_setting("split_map")
        # Flat layout has no split info, so split_map should be absent
        assert split_map is None


# ---------------------------------------------------------------------------
# _import_raw_images
# ---------------------------------------------------------------------------

def _make_raw_images_folder(tmp_path: Path, count: int = 4) -> Path:
    """Create a folder with raw (unannotated) images."""
    folder = tmp_path / "raw_imgs"
    folder.mkdir()
    from PIL import Image
    for i in range(count):
        img = Image.new("RGB", (100, 100), "green")
        img.save(str(folder / f"photo_{i:03d}.jpg"))
    return folder


class TestImportRawImages:
    @pytest.fixture
    def workspace(self, tmp_path):
        pdir = tmp_path / "workspace"
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        store = VerificationStore(pdir)
        return ds, store

    def test_import_saves_images_without_detection(self, tmp_path, workspace):
        """Images are imported with empty detections (detection deferred to review)."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=3)

        import unittest.mock as mock

        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count = _import_raw_images(ds, store, folder)

        assert img_count == 3

        staged = ds.staged_images()
        assert len(staged) == 3

        for img in staged:
            iv = store.get(img.name)
            assert iv is not None
            assert iv.fully_reviewed is False
            assert len(iv.detections) == 0

    def test_import_empty_folder(self, tmp_path, workspace):
        ds, store = workspace
        folder = tmp_path / "empty_raw"
        folder.mkdir()

        import unittest.mock as mock

        progress_mock = mock.MagicMock()
        with mock.patch("pyzm.train.local_import.st") as st_mock:
            st_mock.progress.return_value = progress_mock

            img_count = _import_raw_images(ds, store, folder)

        assert img_count == 0


# ---------------------------------------------------------------------------
# import_raw_images (non-Streamlit, direct function)
# ---------------------------------------------------------------------------

class TestImportRawImagesDirect:
    """Test the public import_raw_images() function directly (no Streamlit)."""

    @pytest.fixture
    def workspace(self, tmp_path):
        pdir = tmp_path / "workspace"
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        store = VerificationStore(pdir)
        return ds, store

    def test_import_basic(self, tmp_path, workspace):
        """Images are imported with empty detections and fully_reviewed=False."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=4)

        img_count = import_raw_images(ds, store, folder)

        assert img_count == 4
        staged = ds.staged_images()
        assert len(staged) == 4

        for img in staged:
            iv = store.get(img.name)
            assert iv is not None
            assert iv.fully_reviewed is False
            assert len(iv.detections) == 0

    def test_import_empty_folder(self, tmp_path, workspace):
        ds, store = workspace
        folder = tmp_path / "empty"
        folder.mkdir()

        img_count = import_raw_images(ds, store, folder)
        assert img_count == 0

    def test_max_images_limits_import(self, tmp_path, workspace):
        """max_images parameter limits the number of imported images."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=10)

        img_count = import_raw_images(ds, store, folder, max_images=3)

        assert img_count == 3
        assert len(ds.staged_images()) == 3

    def test_max_images_zero_imports_all(self, tmp_path, workspace):
        """max_images=0 (default) imports all images."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=5)

        img_count = import_raw_images(ds, store, folder, max_images=0)
        assert img_count == 5

    def test_max_images_larger_than_total(self, tmp_path, workspace):
        """max_images larger than available images imports all."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=3)

        img_count = import_raw_images(ds, store, folder, max_images=100)
        assert img_count == 3

    def test_progress_callback(self, tmp_path, workspace):
        """progress_callback is invoked for each imported image."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=3)

        calls = []
        import_raw_images(
            ds, store, folder,
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    def test_verification_store_persisted(self, tmp_path, workspace):
        """Verification data is saved to disk after import."""
        ds, store = workspace
        folder = _make_raw_images_folder(tmp_path, count=2)

        import_raw_images(ds, store, folder)

        # Reload from disk to verify persistence
        store2 = VerificationStore(ds.project_dir)
        staged = ds.staged_images()
        for img in staged:
            iv = store2.get(img.name)
            assert iv is not None


# ---------------------------------------------------------------------------
# import_local_dataset (non-Streamlit, direct function)
# ---------------------------------------------------------------------------

class TestImportLocalDatasetDirect:
    """Test the public import_local_dataset() function directly (no Streamlit)."""

    @pytest.fixture
    def workspace(self, tmp_path):
        pdir = tmp_path / "workspace"
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        store = VerificationStore(pdir)
        return ds, store

    def test_import_flat(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        img_count, det_count = import_local_dataset(
            ds, store, splits, names_map,
        )

        assert img_count == 3
        assert det_count == 6

    def test_max_images(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        img_count, det_count = import_local_dataset(
            ds, store, splits, names_map, max_images=2,
        )

        assert img_count == 2
        assert det_count == 4  # 2 detections per image

    def test_progress_callback(self, tmp_path, workspace):
        ds, store = workspace
        folder = _make_yolo_folder(tmp_path, flat=True)
        names_map = {0: "person", 1: "car"}
        splits = [(folder / "images", folder / "labels")]

        calls = []
        import_local_dataset(
            ds, store, splits, names_map,
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )

        assert len(calls) == 3
        assert calls[-1] == (3, 3)
