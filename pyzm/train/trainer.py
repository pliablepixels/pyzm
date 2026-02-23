"""Ultralytics YOLO training wrapper.

Handles model loading, hardware detection, training with progress callbacks,
evaluation, and ONNX export.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("pyzm.train")


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    device: str           # "cuda:0", "mps", "cpu"
    gpu_name: str | None  # e.g. "NVIDIA GTX 1050 Ti", "Apple M2 Max"
    vram_gb: float        # 0.0 for CPU
    suggested_batch: int  # Based on VRAM

    @property
    def display(self) -> str:
        if self.gpu_name:
            if self.vram_gb > 0:
                return f"GPU: {self.gpu_name} ({self.vram_gb:.1f}GB)"
            return f"GPU: {self.gpu_name}"
        return "CPU"


@dataclass
class TrainProgress:
    """Training progress snapshot."""

    epoch: int = 0
    total_epochs: int = 0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    message: str = ""
    finished: bool = False
    error: str | None = None


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics."""

    precision: float = 0.0
    recall: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0


@dataclass
class TrainResult:
    """Final training results."""

    best_model: Path | None = None
    last_model: Path | None = None
    best_epoch: int = 0
    final_mAP50: float = 0.0
    final_mAP50_95: float = 0.0
    total_epochs: int = 0
    elapsed_seconds: float = 0.0
    model_size_mb: float = 0.0
    log_lines: list[str] = field(default_factory=list)
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    @classmethod
    def from_disk(cls, project_dir: Path) -> TrainResult | None:
        """Reconstruct a TrainResult from a previous training run on disk.

        Returns None if no completed training run exists.
        """
        import csv

        train_dir = project_dir / "runs" / "train"
        best_pt = train_dir / "weights" / "best.pt"
        csv_path = train_dir / "results.csv"
        if not best_pt.exists() or not csv_path.exists():
            return None

        # Parse results.csv for metrics
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [n.strip() for n in (reader.fieldnames or [])]
                total_epochs = 0
                best_epoch = 0
                best_map50 = -1.0
                final_map50 = 0.0
                final_map50_95 = 0.0
                for row in reader:
                    epoch_str = row.get("epoch", "").strip()
                    if not epoch_str:
                        continue
                    total_epochs = int(epoch_str) + 1  # 0-based -> 1-based

                    map50_val = 0.0
                    map50_95_val = 0.0
                    for key in row:
                        if "mAP50(B)" in key and "mAP50-95" not in key:
                            map50_val = float(row[key].strip())
                        elif "mAP50-95(B)" in key:
                            map50_95_val = float(row[key].strip())

                    final_map50 = map50_val
                    final_map50_95 = map50_95_val
                    if map50_val > best_map50:
                        best_map50 = map50_val
                        best_epoch = total_epochs
        except Exception:
            logger.debug("Could not parse results.csv for from_disk", exc_info=True)
            return None

        if total_epochs == 0:
            return None

        last_pt = train_dir / "weights" / "last.pt"
        model_size = best_pt.stat().st_size / (1024 * 1024)

        return cls(
            best_model=best_pt,
            last_model=last_pt if last_pt.exists() else None,
            best_epoch=best_epoch,
            final_mAP50=final_map50,
            final_mAP50_95=final_map50_95,
            total_epochs=total_epochs,
            model_size_mb=model_size,
        )


class YOLOTrainer:
    """Wraps Ultralytics YOLO for fine-tuning.

    Parameters
    ----------
    base_model : str
        Model name (e.g. ``"yolo11s"``) or path to a ``.pt`` file.
        Plain names auto-download from Ultralytics hub.
    project_dir : Path
        Project directory (contains dataset.yaml, runs/).
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda:0"`` etc.
    """

    def __init__(
        self,
        base_model: str,
        project_dir: Path,
        device: str = "auto",
    ) -> None:
        self.base_model = base_model
        self.project_dir = Path(project_dir)
        self.device = device
        self._model: Any = None  # ultralytics.YOLO
        self._stop_event = threading.Event()

    def has_checkpoint(self) -> bool:
        """Return True if a resumable ``last.pt`` checkpoint exists."""
        return (self.project_dir / "runs" / "train" / "weights" / "last.pt").exists()

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_hardware() -> HardwareInfo:
        """Detect GPU/CPU and suggest training batch size."""
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
                # ~0.5 GB per image at 640px; cap at 64 for high-VRAM GPUs
                suggested = max(4, min(64, int(vram_gb * 2)))
                return HardwareInfo(
                    device="cuda:0",
                    gpu_name=props.name,
                    vram_gb=vram_gb,
                    suggested_batch=suggested,
                )

            if torch.backends.mps.is_available():
                # Apple Silicon – unified memory, no separate VRAM query
                import subprocess, platform
                try:
                    chip = subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        text=True,
                    ).strip()
                except Exception:
                    chip = platform.processor() or "Apple Silicon"
                return HardwareInfo(
                    device="mps",
                    gpu_name=chip,
                    vram_gb=0.0,
                    suggested_batch=16,
                )
        except (ImportError, Exception):
            pass

        return HardwareInfo(
            device="cpu",
            gpu_name=None,
            vram_gb=0.0,
            suggested_batch=4,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the Ultralytics YOLO model."""
        from ultralytics import YOLO

        model_path = self.base_model
        # If it doesn't look like a file path, assume it's a hub name
        if not Path(model_path).suffix:
            model_path = f"{model_path}.pt"

        # If already an absolute/existing file, use as-is.
        # Otherwise pre-download into project_dir so Ultralytics doesn't
        # pollute the working directory with .pt files.
        if not Path(model_path).exists():
            dest = self.project_dir / Path(model_path).name
            if not dest.exists():
                from ultralytics.utils.downloads import attempt_download_asset
                # Download into project_dir; attempt_download_asset may
                # ignore the directory component, so pass the bare
                # filename and use a cwd guard as a fallback.
                import os
                original_cwd = os.getcwd()
                os.chdir(self.project_dir)
                try:
                    attempt_download_asset(Path(model_path).name)
                finally:
                    os.chdir(original_cwd)
            model_path = str(dest)

        self._model = YOLO(model_path)
        return self._model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_yaml: Path,
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        resume: bool = False,
        progress_callback: Callable[[TrainProgress], None] | None = None,
    ) -> TrainResult:
        """Run fine-tuning.

        This is intended to be called from a background thread.
        Use :meth:`request_stop` to signal early stopping.

        Parameters
        ----------
        dataset_yaml:
            Path to the YOLO dataset.yaml.
        epochs:
            Number of training epochs.
        batch:
            Batch size.
        imgsz:
            Training image size.
        resume:
            If True and a ``last.pt`` checkpoint exists, resume training
            from that checkpoint instead of starting from scratch.
        progress_callback:
            Called after each epoch with a :class:`TrainProgress` snapshot.
        """
        import time

        self._stop_event.clear()

        # Resume from checkpoint if requested
        resume_training = False
        if resume:
            last_pt = self.project_dir / "runs" / "train" / "weights" / "last.pt"
            if last_pt.exists():
                from ultralytics import YOLO
                self._model = YOLO(str(last_pt))
                resume_training = True
                logger.info("Resuming training from %s", last_pt)
            else:
                logger.warning("Resume requested but no last.pt found, starting fresh")

        if not resume_training:
            self._load_model()
        model = self._model

        device = self.device
        if device == "auto":
            hw = self.detect_hardware()
            device = hw.device

        runs_dir = self.project_dir / "runs"

        progress = TrainProgress(total_epochs=epochs)
        final_metrics: dict[str, Any] | None = None

        def _on_train_epoch_end(trainer: Any) -> None:
            """Ultralytics callback after each training epoch."""
            if self._stop_event.is_set():
                raise KeyboardInterrupt("Training stopped by user")

            progress.epoch = trainer.epoch + 1
            metrics = trainer.metrics or {}
            progress.box_loss = float(trainer.loss_items[0]) if trainer.loss_items is not None else 0.0
            progress.cls_loss = float(trainer.loss_items[1]) if trainer.loss_items is not None and len(trainer.loss_items) > 1 else 0.0
            progress.mAP50 = float(metrics.get("metrics/mAP50(B)", 0.0))
            progress.mAP50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
            progress.message = (
                f"Epoch {progress.epoch}/{epochs} | "
                f"mAP50={progress.mAP50:.3f}"
            )
            if progress_callback:
                progress_callback(progress)

        model.add_callback("on_train_epoch_end", _on_train_epoch_end)

        def _on_train_end(trainer: Any) -> None:
            """Capture final eval metrics from best.pt validation."""
            nonlocal final_metrics
            vm = getattr(trainer, "validator", None)
            if vm is None:
                return
            m = getattr(vm, "metrics", None)
            if m is None:
                return

            # Use results_dict (stable across Ultralytics versions);
            # map50/map properties were removed in 8.4.x.
            rd = getattr(m, "results_dict", None) or {}
            final_metrics = {
                "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
                "map": float(rd.get("metrics/mAP50-95(B)", 0.0)),
                "per_class": {},
            }
            names = getattr(m, "names", {}) or getattr(trainer.model, "names", {})
            box = getattr(m, "box", None)
            if box is not None:
                # ap_class_index holds class IDs; class_result() takes
                # an array position (changed in Ultralytics 8.4.x).
                for pos, cls_id in enumerate(box.ap_class_index):
                    p, r, ap50, ap = m.class_result(pos)
                    cls_name = names.get(int(cls_id), f"class_{cls_id}")
                    final_metrics["per_class"][cls_name] = ClassMetrics(
                        precision=float(p),
                        recall=float(r),
                        ap50=float(ap50),
                        ap50_95=float(ap),
                    )

        model.add_callback("on_train_end", _on_train_end)

        # Run training from the project directory so Ultralytics
        # side-effects (e.g. AMP check downloading yolo26n.pt) don't
        # pollute the user's working directory.
        import os
        original_cwd = os.getcwd()
        os.chdir(self.project_dir)

        start = time.monotonic()
        try:
            train_kwargs: dict[str, Any] = dict(
                data=str(dataset_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                project=str(runs_dir),
                name="train",
                exist_ok=True,
                verbose=True,
            )
            if resume_training:
                train_kwargs["resume"] = True
            results = model.train(**train_kwargs)
        except KeyboardInterrupt:
            logger.info("Training stopped by user")
            progress.finished = True
            progress.message = "Training stopped by user"
            if progress_callback:
                progress_callback(progress)
        except Exception as exc:
            progress.error = str(exc)
            progress.finished = True
            if progress_callback:
                progress_callback(progress)
            raise
        else:
            progress.finished = True
            progress.message = "Training complete"
            if progress_callback:
                progress_callback(progress)
        finally:
            os.chdir(original_cwd)

        elapsed = time.monotonic() - start

        # Locate output weights
        weights_dir = runs_dir / "train" / "weights"
        best_pt = weights_dir / "best.pt" if weights_dir.exists() else None
        last_pt = weights_dir / "last.pt" if weights_dir.exists() else None
        if best_pt and not best_pt.exists():
            best_pt = None
        if last_pt and not last_pt.exists():
            last_pt = None

        model_size = best_pt.stat().st_size / (1024 * 1024) if best_pt else 0.0

        # Use final_metrics (from on_train_end) if available, else fall back
        # to last epoch's progress values.
        if final_metrics:
            map50 = final_metrics["map50"]
            map50_95 = final_metrics["map"]
            per_class = final_metrics["per_class"]
        else:
            map50 = progress.mAP50
            map50_95 = progress.mAP50_95
            per_class = {}

        best_epoch = self._read_best_epoch(runs_dir / "train")

        return TrainResult(
            best_model=best_pt,
            last_model=last_pt,
            best_epoch=best_epoch if best_epoch > 0 else progress.epoch,
            final_mAP50=map50,
            final_mAP50_95=map50_95,
            total_epochs=progress.epoch,
            elapsed_seconds=elapsed,
            model_size_mb=model_size,
            per_class=per_class,
        )

    def request_stop(self) -> None:
        """Signal the training loop to stop after the current epoch."""
        self._stop_event.set()

    @staticmethod
    def _read_best_epoch(train_dir: Path) -> int:
        """Read results.csv and return the 1-based epoch with highest mAP50.

        Returns 0 if the file can't be read.
        """
        import csv

        csv_path = train_dir / "results.csv"
        if not csv_path.exists():
            return 0
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
                best_epoch = 0
                best_map50 = -1.0
                for row in reader:
                    epoch_str = row.get("epoch", "").strip()
                    map50_str = ""
                    for key in row:
                        if "mAP50(B)" in key and "mAP50-95" not in key:
                            map50_str = row[key].strip()
                            break
                    if not epoch_str or not map50_str:
                        continue
                    epoch_val = int(epoch_str) + 1  # CSV is 0-based, we display 1-based
                    map50_val = float(map50_str)
                    if map50_val > best_map50:
                        best_map50 = map50_val
                        best_epoch = epoch_val
                return best_epoch
        except Exception:
            logger.debug("Could not parse results.csv for best epoch", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, image: "Any") -> list[dict]:
        """Run the fine-tuned model on a single image.

        Parameters
        ----------
        image:
            Image path (str) or numpy array.

        Returns
        -------
        List of dicts with keys: label, confidence, bbox (x1,y1,x2,y2).
        """
        if self._model is None:
            # Try loading best.pt from project
            best = self.project_dir / "runs" / "train" / "weights" / "best.pt"
            if best.exists():
                from ultralytics import YOLO
                self._model = YOLO(str(best))
            else:
                raise FileNotFoundError("No trained model found. Run training first.")

        results = self._model(image, verbose=False)
        detections: list[dict] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id] if cls_id in r.names else str(cls_id)
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
        return detections

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, output_path: Path | None = None) -> Path:
        """Export the best trained model to ONNX format.

        Parameters
        ----------
        output_path:
            Full destination file path (e.g. ``/path/to/model.onnx``).
            Defaults to ``runs/train/weights/best.onnx`` in the project dir.

        Returns
        -------
        Path to the exported ``.onnx`` file.
        """
        best_pt = self.project_dir / "runs" / "train" / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"No best.pt found at {best_pt}")

        from ultralytics import YOLO

        model = YOLO(str(best_pt))
        onnx_path = model.export(format="onnx")
        onnx_path = Path(onnx_path)

        if output_path and Path(output_path) != onnx_path:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_path, dest)
            return dest

        return onnx_path
