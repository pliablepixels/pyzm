"""Tests for pyzm.train.trainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyzm.train.trainer import (
    ClassMetrics,
    HardwareInfo,
    TrainProgress,
    TrainResult,
    YOLOTrainer,
    adaptive_finetune_params,
)


# ---------------------------------------------------------------------------
# adaptive_finetune_params
# ---------------------------------------------------------------------------

class TestAdaptiveFinetuneParams:
    """Test dataset-size-adaptive hyperparameter selection.

    Every tier must freeze backbone layers and use cosine LR to ensure
    fine-tuning never overwrites pretrained feature representations.
    Augmentation varies by mode (new_class vs refine).
    """

    _EXPECTED_KEYS = {
        "freeze", "lr0", "patience", "cos_lr", "val_ratio", "tier",
        "mosaic", "erasing", "scale", "mixup", "close_mosaic", "mode",
    }

    def test_expected_keys(self):
        result = adaptive_finetune_params(100)
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_default_mode_is_new_class(self):
        assert adaptive_finetune_params(100)["mode"] == "new_class"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            adaptive_finetune_params(100, mode="bad")

    # --- new_class mode (default) ---

    def test_small_new_class(self):
        p = adaptive_finetune_params(50)
        assert p["tier"] == "small"
        assert p["freeze"] == 10
        assert p["lr0"] == pytest.approx(0.0005)
        assert p["patience"] == 10
        assert p["cos_lr"] is True
        assert p["val_ratio"] == pytest.approx(0.15)
        assert p["mosaic"] == pytest.approx(0.0)
        assert p["erasing"] == pytest.approx(0.0)

    def test_medium_new_class(self):
        p = adaptive_finetune_params(500)
        assert p["tier"] == "medium"
        assert p["freeze"] == 10
        assert p["lr0"] == pytest.approx(0.001)
        assert p["mosaic"] == pytest.approx(0.0)
        assert p["erasing"] == pytest.approx(0.0)

    def test_large_new_class(self):
        p = adaptive_finetune_params(2000)
        assert p["tier"] == "large"
        assert p["freeze"] == 5
        assert p["lr0"] == pytest.approx(0.002)
        assert p["mosaic"] == pytest.approx(0.3)
        assert p["erasing"] == pytest.approx(0.1)

    def test_xlarge_new_class(self):
        p = adaptive_finetune_params(10000)
        assert p["tier"] == "xlarge"
        assert p["freeze"] == 3
        assert p["lr0"] == pytest.approx(0.005)
        assert p["mosaic"] == pytest.approx(0.5)
        assert p["erasing"] == pytest.approx(0.15)

    # --- refine mode ---

    def test_small_refine(self):
        p = adaptive_finetune_params(50, mode="refine")
        assert p["tier"] == "small"
        assert p["mode"] == "refine"
        assert p["mosaic"] == pytest.approx(0.3)
        assert p["erasing"] == pytest.approx(0.1)
        # Base params unchanged by mode
        assert p["freeze"] == 10
        assert p["lr0"] == pytest.approx(0.0005)

    def test_large_refine(self):
        p = adaptive_finetune_params(2000, mode="refine")
        assert p["tier"] == "large"
        assert p["mosaic"] == pytest.approx(0.7)
        assert p["erasing"] == pytest.approx(0.2)

    def test_xlarge_refine(self):
        p = adaptive_finetune_params(10000, mode="refine")
        assert p["tier"] == "xlarge"
        assert p["mosaic"] == pytest.approx(0.8)
        assert p["mixup"] == pytest.approx(0.05)

    # --- boundaries ---

    def test_boundary_199_is_small(self):
        assert adaptive_finetune_params(199)["tier"] == "small"

    def test_boundary_200_is_medium(self):
        assert adaptive_finetune_params(200)["tier"] == "medium"

    def test_boundary_999_is_medium(self):
        assert adaptive_finetune_params(999)["tier"] == "medium"

    def test_boundary_1000_is_large(self):
        assert adaptive_finetune_params(1000)["tier"] == "large"

    def test_boundary_4999_is_large(self):
        assert adaptive_finetune_params(4999)["tier"] == "large"

    def test_boundary_5000_is_xlarge(self):
        assert adaptive_finetune_params(5000)["tier"] == "xlarge"

    def test_minimal_dataset(self):
        p = adaptive_finetune_params(1)
        assert p["tier"] == "small"
        assert p["freeze"] == 10

    # --- invariants (must hold for ALL tiers and modes) ---

    _ALL_SIZES = (1, 50, 199, 200, 500, 999, 1000, 2000, 4999, 5000, 50000)

    def test_all_tiers_freeze_backbone(self):
        """Fine-tuning must always freeze some backbone layers."""
        for n in self._ALL_SIZES:
            for m in ("new_class", "refine"):
                p = adaptive_finetune_params(n, mode=m)
                assert p["freeze"] > 0, f"n={n} mode={m} has freeze=0"

    def test_all_tiers_use_cosine_lr(self):
        """Cosine annealing must always be enabled."""
        for n in self._ALL_SIZES:
            for m in ("new_class", "refine"):
                p = adaptive_finetune_params(n, mode=m)
                assert p["cos_lr"] is True, f"n={n} mode={m} has cos_lr=False"

    def test_new_class_never_exceeds_refine_augmentation(self):
        """new_class augmentation must be <= refine for the same tier."""
        for n in self._ALL_SIZES:
            nc = adaptive_finetune_params(n, mode="new_class")
            rf = adaptive_finetune_params(n, mode="refine")
            assert nc["mosaic"] <= rf["mosaic"], f"n={n} mosaic"
            assert nc["erasing"] <= rf["erasing"], f"n={n} erasing"
            assert nc["mixup"] <= rf["mixup"], f"n={n} mixup"

    def test_small_new_class_has_no_mosaic(self):
        """Small new-class datasets must not use mosaic at all."""
        for n in (1, 50, 100, 199):
            p = adaptive_finetune_params(n, mode="new_class")
            assert p["mosaic"] == 0.0, f"n={n} has mosaic={p['mosaic']}"


# ---------------------------------------------------------------------------
# HardwareInfo
# ---------------------------------------------------------------------------

class TestHardwareInfo:
    def test_gpu_display(self):
        hw = HardwareInfo(
            device="cuda:0", gpu_name="NVIDIA GTX 1050 Ti",
            vram_gb=4.0, suggested_batch=16,
        )
        assert "NVIDIA GTX 1050 Ti" in hw.display
        assert "4.0GB" in hw.display

    def test_cpu_display(self):
        hw = HardwareInfo(
            device="cpu", gpu_name=None, vram_gb=0.0, suggested_batch=4,
        )
        assert hw.display == "CPU"


# ---------------------------------------------------------------------------
# TrainProgress
# ---------------------------------------------------------------------------

class TestTrainProgress:
    def test_defaults(self):
        p = TrainProgress()
        assert p.epoch == 0
        assert p.finished is False
        assert p.error is None


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

class TestClassMetrics:
    def test_defaults(self):
        cm = ClassMetrics()
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.ap50 == 0.0
        assert cm.ap50_95 == 0.0

    def test_custom_values(self):
        cm = ClassMetrics(precision=0.9, recall=0.8, ap50=0.85, ap50_95=0.7)
        assert cm.precision == pytest.approx(0.9)
        assert cm.ap50_95 == pytest.approx(0.7)


class TestTrainResult:
    def test_defaults(self):
        r = TrainResult()
        assert r.best_model is None
        assert r.final_mAP50 == 0.0
        assert r.model_size_mb == 0.0
        assert r.per_class == {}

    def test_per_class_field(self):
        r = TrainResult(
            per_class={
                "cat": ClassMetrics(precision=0.9, recall=0.85, ap50=0.88, ap50_95=0.7),
                "dog": ClassMetrics(precision=0.8, recall=0.75, ap50=0.82, ap50_95=0.6),
            },
        )
        assert len(r.per_class) == 2
        assert r.per_class["cat"].ap50 == pytest.approx(0.88)
        assert r.per_class["dog"].recall == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# YOLOTrainer
# ---------------------------------------------------------------------------

class TestYOLOTrainer:
    def test_init(self, tmp_path: Path):
        trainer = YOLOTrainer(
            base_model="yolo11s",
            project_dir=tmp_path / "proj",
            device="cpu",
        )
        assert trainer.base_model == "yolo11s"
        assert trainer.device == "cpu"

    def test_detect_hardware_cpu_fallback(self):
        """When torch is not available or no CUDA, falls back to CPU."""
        with patch.dict("sys.modules", {"torch": None}):
            hw = YOLOTrainer.detect_hardware()
            assert hw.device == "cpu"
            assert hw.gpu_name is None
            assert hw.suggested_batch == 4

    def test_detect_hardware_with_cuda(self):
        """Mock torch.cuda to test GPU detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.name = "NVIDIA Test GPU"
        mock_props.total_memory = 8 * (1024 ** 3)  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from pyzm.train.trainer import YOLOTrainer as T
            hw = T.detect_hardware()
            assert hw.device == "cuda:0"
            assert hw.gpu_name == "NVIDIA Test GPU"
            assert hw.vram_gb == pytest.approx(8.0)
            assert hw.suggested_batch >= 4

    def test_load_model_adds_pt_extension(self, tmp_path: Path):
        """A plain model name like 'yolo11s' gets .pt appended and downloaded to project_dir."""
        trainer = YOLOTrainer("yolo11s", tmp_path)
        mock_yolo = MagicMock(return_value=MagicMock())

        def _fake_download(dest):
            Path(dest).touch()
            return dest

        mock_downloads = MagicMock(attempt_download_asset=_fake_download)
        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(YOLO=mock_yolo),
            "ultralytics.utils": MagicMock(),
            "ultralytics.utils.downloads": mock_downloads,
        }):
            trainer._load_model()

        mock_yolo.assert_called_once_with(str(tmp_path / "yolo11s.pt"))

    def test_export_onnx_no_model_raises(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        with pytest.raises(FileNotFoundError, match="No best.pt"):
            trainer.export_onnx()

    def test_evaluate_no_model_raises(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        with pytest.raises(FileNotFoundError, match="No trained model"):
            trainer.evaluate("dummy.jpg")

    def test_request_stop(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        assert not trainer._stop_event.is_set()
        trainer.request_stop()
        assert trainer._stop_event.is_set()

    def test_export_onnx_copies_to_output_dir(self, tmp_path: Path):
        """When best.pt exists, export should work (mocked)."""
        proj = tmp_path / "proj"
        weights_dir = proj / "runs" / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake model data")

        trainer = YOLOTrainer("yolo11s", proj)
        dest_path = tmp_path / "deploy" / "my_model.onnx"

        mock_model = MagicMock()
        onnx_out = weights_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx data")
        mock_model.export.return_value = str(onnx_out)

        mock_yolo = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            result = trainer.export_onnx(output_path=dest_path)

        assert result == dest_path
        assert result.exists()

    def test_evaluate_with_trained_model(self, tmp_path: Path):
        """Test evaluate loads best.pt and parses results."""
        proj = tmp_path / "proj"
        weights_dir = proj / "runs" / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake")

        trainer = YOLOTrainer("yolo11s", proj)

        # Mock YOLO model and its output
        mock_box = MagicMock()
        mock_box.xyxy.__getitem__ = lambda self, idx: MagicMock(
            tolist=lambda: [10.0, 20.0, 100.0, 200.0]
        )
        mock_box.cls.__getitem__ = lambda self, idx: 0
        mock_box.conf.__getitem__ = lambda self, idx: 0.95

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        mock_yolo = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            dets = trainer.evaluate("test.jpg")

        assert len(dets) == 1
        assert dets[0]["label"] == "person"
        assert dets[0]["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _read_best_epoch
# ---------------------------------------------------------------------------

class TestReadBestEpoch:
    def test_no_csv_returns_zero(self, tmp_path: Path):
        assert YOLOTrainer._read_best_epoch(tmp_path) == 0

    def test_parses_best_epoch(self, tmp_path: Path):
        csv_content = (
            "                  epoch,      train/box_loss,      train/cls_loss,       metrics/mAP50(B),  metrics/mAP50-95(B)\n"
            "                      0,             1.5000,             2.0000,              0.100,              0.050\n"
            "                      1,             1.2000,             1.5000,              0.800,              0.400\n"
            "                      2,             1.0000,             1.2000,              0.600,              0.350\n"
        )
        (tmp_path / "results.csv").write_text(csv_content)
        # Epoch 1 (0-based) has highest mAP50 (0.800), so best_epoch = 2 (1-based)
        assert YOLOTrainer._read_best_epoch(tmp_path) == 2

    def test_malformed_csv_returns_zero(self, tmp_path: Path):
        (tmp_path / "results.csv").write_text("not,a,valid,csv\nfoo,bar,baz,qux\n")
        assert YOLOTrainer._read_best_epoch(tmp_path) == 0

    def test_parses_csv_without_leading_spaces(self, tmp_path: Path):
        """CSV with clean (no leading spaces) column names should also work."""
        csv_content = (
            "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "0,1.5000,2.0000,0.100,0.050\n"
            "1,1.2000,1.5000,0.900,0.500\n"
            "2,1.0000,1.2000,0.600,0.350\n"
        )
        (tmp_path / "results.csv").write_text(csv_content)
        assert YOLOTrainer._read_best_epoch(tmp_path) == 2


# ---------------------------------------------------------------------------
# Batch size suggestions
# ---------------------------------------------------------------------------

class TestBatchSizeSuggestion:
    """Test the batch-size heuristic at various VRAM levels."""

    @pytest.mark.parametrize("vram_gb,expected", [
        (2.0, 4),    # 2 * 2 = 4, clamped to min 4
        (4.0, 8),    # 4 * 2 = 8
        (8.0, 16),   # 8 * 2 = 16
        (24.0, 48),  # 24 * 2 = 48, under cap of 64
    ])
    def test_batch_suggestions(self, vram_gb: float, expected: int):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = int(vram_gb * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from pyzm.train.trainer import YOLOTrainer as T
            hw = T.detect_hardware()
            assert hw.suggested_batch == expected


# ---------------------------------------------------------------------------
# has_checkpoint
# ---------------------------------------------------------------------------

class TestHasCheckpoint:
    def test_no_checkpoint(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        assert trainer.has_checkpoint() is False

    def test_with_checkpoint(self, tmp_path: Path):
        weights_dir = tmp_path / "runs" / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "last.pt").write_bytes(b"fake checkpoint")
        trainer = YOLOTrainer("yolo11s", tmp_path)
        assert trainer.has_checkpoint() is True
