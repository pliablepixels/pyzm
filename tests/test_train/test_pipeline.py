"""Tests for pyzm.train.pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pyzm.train.pipeline import run_correct_pipeline, run_pipeline
from pyzm.train.trainer import HardwareInfo, TrainResult
from pyzm.train.verification import DetectionStatus, VerifiedDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_folder(tmp_path: Path) -> Path:
    """Create a minimal YOLO dataset folder for pipeline tests."""
    folder = tmp_path / "yolo_ds"
    folder.mkdir()
    (folder / "data.yaml").write_text("names: {0: person, 1: car}\n")

    img_dir = folder / "images"
    lbl_dir = folder / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    for i in range(3):
        img = Image.new("RGB", (100, 100), "red")
        img.save(str(img_dir / f"img{i:03d}.jpg"))
        (lbl_dir / f"img{i:03d}.txt").write_text(
            "0 0.5 0.5 0.2 0.3\n1 0.7 0.7 0.1 0.1\n"
        )

    return folder


@pytest.fixture
def mock_trainer():
    """Patch YOLOTrainer in the pipeline module with sane defaults."""
    with patch("pyzm.train.pipeline.YOLOTrainer") as MockTrainer:
        instance = MagicMock()
        MockTrainer.return_value = instance
        MockTrainer.detect_hardware.return_value = HardwareInfo(
            device="cpu", gpu_name=None, vram_gb=0.0, suggested_batch=4,
        )
        instance.train.return_value = TrainResult(
            best_model=None,
            total_epochs=10,
            elapsed_seconds=10.0,
        )
        yield MockTrainer, instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_rejects_missing_data_yaml(self, tmp_path):
        """Validation rejects a path without data.yaml."""
        folder = tmp_path / "empty_ds"
        folder.mkdir()

        with pytest.raises(ValueError, match="Missing data.yaml"):
            run_pipeline(folder, workspace_dir=tmp_path / "ws")

    def test_rejects_nonexistent_path(self, tmp_path):
        """Validation rejects a non-existent path."""
        with pytest.raises(ValueError, match="Not a directory"):
            run_pipeline(tmp_path / "nonexistent", workspace_dir=tmp_path / "ws")

    def test_full_flow(self, tmp_path, mock_trainer):
        """Full pipeline: validate -> import -> split -> train -> export."""
        MockTrainer, trainer_inst = mock_trainer
        dataset_path = _make_yolo_folder(tmp_path)
        workspace = tmp_path / "workspace"

        fake_best_pt = tmp_path / "best.pt"
        fake_best_pt.write_bytes(b"fake")
        fake_onnx = tmp_path / "model.onnx"

        trainer_inst.train.return_value = TrainResult(
            best_model=fake_best_pt,
            best_epoch=10,
            final_mAP50=0.85,
            final_mAP50_95=0.65,
            total_epochs=50,
            elapsed_seconds=120.0,
            model_size_mb=12.5,
        )
        trainer_inst.export_onnx.return_value = fake_onnx

        result = run_pipeline(
            dataset_path,
            workspace_dir=workspace,
            project_name="test_proj",
            model="yolo11s",
            epochs=50,
            batch=8,
            imgsz=640,
            val_ratio=0.2,
        )

        assert result.final_mAP50 == pytest.approx(0.85)
        assert result.best_epoch == 10

        # Trainer constructed with correct args
        MockTrainer.assert_called_once_with(
            "yolo11s", workspace / "test_proj", device="auto",
        )

        # train() called with correct parameters
        train_call = trainer_inst.train.call_args
        assert train_call.kwargs["epochs"] == 50
        assert train_call.kwargs["batch"] == 8
        assert train_call.kwargs["imgsz"] == 640

        # export_onnx was called since best_model exists
        trainer_inst.export_onnx.assert_called_once_with(None)

    def test_creates_project_directory(self, tmp_path, mock_trainer):
        """Pipeline creates the project directory structure."""
        dataset_path = _make_yolo_folder(tmp_path)
        workspace = tmp_path / "workspace"

        run_pipeline(
            dataset_path,
            workspace_dir=workspace,
            project_name="my_project",
        )

        project_dir = workspace / "my_project"
        assert project_dir.is_dir()
        assert (project_dir / "images" / "all").is_dir()
        assert (project_dir / "labels" / "all").is_dir()
        assert (project_dir / "dataset.yaml").exists()

    def test_skips_export_when_no_best_model(self, tmp_path, mock_trainer):
        """When training produces no best_model, ONNX export is skipped."""
        _, trainer_inst = mock_trainer
        dataset_path = _make_yolo_folder(tmp_path)

        run_pipeline(dataset_path, workspace_dir=tmp_path / "ws")

        trainer_inst.export_onnx.assert_not_called()

    def test_passes_custom_parameters(self, tmp_path, mock_trainer):
        """All parameters flow through to the trainer correctly."""
        MockTrainer, trainer_inst = mock_trainer
        dataset_path = _make_yolo_folder(tmp_path)
        workspace = tmp_path / "workspace"
        output_path = tmp_path / "output" / "model.onnx"

        fake_best = tmp_path / "best.pt"
        fake_best.write_bytes(b"fake")
        trainer_inst.train.return_value = TrainResult(
            best_model=fake_best,
            total_epochs=100,
            elapsed_seconds=600.0,
            best_epoch=80,
            final_mAP50=0.9,
            final_mAP50_95=0.7,
            model_size_mb=25.0,
        )
        trainer_inst.export_onnx.return_value = output_path

        run_pipeline(
            dataset_path,
            workspace_dir=workspace,
            model="yolo11m",
            epochs=100,
            batch=32,
            imgsz=1280,
            val_ratio=0.3,
            output=output_path,
            device="cuda:0",
        )

        # Trainer constructed with correct model and device
        MockTrainer.assert_called_once_with(
            "yolo11m", workspace / dataset_path.name, device="cuda:0",
        )

        # train() called with correct parameters
        train_call = trainer_inst.train.call_args
        assert train_call.kwargs["epochs"] == 100
        assert train_call.kwargs["batch"] == 32
        assert train_call.kwargs["imgsz"] == 1280

        # export_onnx called with output path
        trainer_inst.export_onnx.assert_called_once_with(output_path)

    def test_auto_batch_from_hardware(self, tmp_path, mock_trainer):
        """When batch=None, pipeline uses hw.suggested_batch."""
        MockTrainer, trainer_inst = mock_trainer
        MockTrainer.detect_hardware.return_value = HardwareInfo(
            device="cuda:0", gpu_name="Test GPU", vram_gb=8.0, suggested_batch=16,
        )

        dataset_path = _make_yolo_folder(tmp_path)
        run_pipeline(dataset_path, workspace_dir=tmp_path / "ws", batch=None)

        train_call = trainer_inst.train.call_args
        assert train_call.kwargs["batch"] == 16

    def test_project_name_defaults_to_folder_name(self, tmp_path, mock_trainer):
        """When project_name is None, uses the dataset folder name."""
        dataset_path = _make_yolo_folder(tmp_path)
        workspace = tmp_path / "workspace"

        run_pipeline(dataset_path, workspace_dir=workspace)

        project_dir = workspace / "yolo_ds"
        assert project_dir.is_dir()


# ---------------------------------------------------------------------------
# Helpers for correct pipeline tests
# ---------------------------------------------------------------------------

def _make_raw_folder(tmp_path: Path, count: int = 3) -> Path:
    """Create a folder with raw images (no data.yaml)."""
    folder = tmp_path / "raw_imgs"
    folder.mkdir()
    for i in range(count):
        img = Image.new("RGB", (100, 100), "green")
        img.save(str(folder / f"photo_{i:03d}.jpg"))
    return folder


@pytest.fixture
def mock_correct_trainer():
    """Patch YOLOTrainer and auto_detect_image for correct pipeline tests."""
    with patch("pyzm.train.pipeline.YOLOTrainer") as MockTrainer:
        instance = MagicMock()
        MockTrainer.return_value = instance
        MockTrainer.detect_hardware.return_value = HardwareInfo(
            device="cpu", gpu_name=None, vram_gb=0.0, suggested_batch=4,
        )
        instance.train.return_value = TrainResult(
            best_model=None,
            total_epochs=10,
            elapsed_seconds=10.0,
        )
        yield MockTrainer, instance


# ---------------------------------------------------------------------------
# Tests for run_correct_pipeline
# ---------------------------------------------------------------------------

class TestRunCorrectPipeline:
    def test_rejects_nonexistent_folder(self, tmp_path):
        """Non-existent folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Not a directory"):
            run_correct_pipeline(
                tmp_path / "nonexistent",
                workspace_dir=tmp_path / "ws",
            )

    def test_rejects_empty_folder(self, tmp_path, mock_correct_trainer):
        """Empty folder raises ValueError."""
        folder = tmp_path / "empty"
        folder.mkdir()

        with patch("pyzm.train.pipeline.import_correct_model", return_value=(0, 0)):
            with pytest.raises(ValueError, match="No images found"):
                run_correct_pipeline(
                    folder,
                    workspace_dir=tmp_path / "ws",
                )

    def test_full_flow(self, tmp_path, mock_correct_trainer):
        """Full correct pipeline: scan -> detect -> approve -> train."""
        MockTrainer, trainer_inst = mock_correct_trainer
        folder = _make_raw_folder(tmp_path)
        workspace = tmp_path / "workspace"

        fake_best_pt = tmp_path / "best.pt"
        fake_best_pt.write_bytes(b"fake")
        trainer_inst.train.return_value = TrainResult(
            best_model=fake_best_pt,
            best_epoch=10,
            final_mAP50=0.75,
            final_mAP50_95=0.55,
            total_epochs=50,
            elapsed_seconds=60.0,
            model_size_mb=10.0,
        )
        trainer_inst.export_onnx.return_value = tmp_path / "model.onnx"

        # Mock import_correct_model to set up store with detections
        from pyzm.train.dataset import Annotation

        def _mock_import(ds, store, folder_, **kwargs):
            for img_path in sorted(folder_.glob("*.jpg")):
                dest = ds.add_image(img_path, [])
                ann = Annotation(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.3)
                det = VerifiedDetection(
                    detection_id="det_0",
                    original=ann,
                    status=DetectionStatus.PENDING,
                    original_label="car",
                    confidence=0.8,
                )
                from pyzm.train.verification import ImageVerification
                store.set(ImageVerification(
                    image_name=dest.name,
                    detections=[det],
                    fully_reviewed=False,
                ))
            store.save()
            cb = kwargs.get("progress_callback")
            return 3, 3

        with patch("pyzm.train.pipeline.import_correct_model", side_effect=_mock_import):
            result = run_correct_pipeline(
                folder,
                workspace_dir=workspace,
                project_name="correct_test",
                epochs=50,
            )

        assert result.final_mAP50 == pytest.approx(0.75)
        trainer_inst.export_onnx.assert_called_once()

    def test_creates_project_directory(self, tmp_path, mock_correct_trainer):
        """Pipeline creates project directory."""
        folder = _make_raw_folder(tmp_path)
        workspace = tmp_path / "workspace"

        from pyzm.train.dataset import Annotation

        def _mock_import(ds, store, folder_, **kwargs):
            for img_path in sorted(folder_.glob("*.jpg")):
                dest = ds.add_image(img_path, [])
                ann = Annotation(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.3)
                det = VerifiedDetection(
                    detection_id="det_0",
                    original=ann,
                    status=DetectionStatus.PENDING,
                    original_label="car",
                )
                from pyzm.train.verification import ImageVerification
                store.set(ImageVerification(
                    image_name=dest.name,
                    detections=[det],
                    fully_reviewed=False,
                ))
            store.save()
            return 3, 3

        with patch("pyzm.train.pipeline.import_correct_model", side_effect=_mock_import):
            run_correct_pipeline(
                folder,
                workspace_dir=workspace,
                project_name="my_correct",
            )

        project_dir = workspace / "my_correct"
        assert project_dir.is_dir()
