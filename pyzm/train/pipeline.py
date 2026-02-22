"""Headless training pipeline for ``python -m pyzm.train <dataset>``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pyzm.train.dataset import YOLODataset
from pyzm.train.local_import import import_local_dataset, validate_yolo_folder
from pyzm.train.trainer import TrainResult, YOLOTrainer
from pyzm.train.verification import VerificationStore

logger = logging.getLogger("pyzm.train")

_DEFAULT_WORKSPACE = Path.home() / ".pyzm" / "training"


def _print_progress(current: int, total: int) -> None:
    pct = current * 100 // total
    sys.stdout.write(f"\rImporting... {current}/{total} ({pct}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def run_pipeline(
    dataset_path: Path,
    *,
    project_name: str | None = None,
    workspace_dir: Path | None = None,
    model: str = "yolo11s",
    epochs: int = 50,
    batch: int | None = None,
    imgsz: int = 640,
    val_ratio: float = 0.2,
    output: Path | None = None,
    device: str = "auto",
    max_per_class: int = 0,
) -> TrainResult:
    """Run the full training pipeline headlessly.

    Steps: validate → import → split → train → export.

    Parameters
    ----------
    dataset_path:
        Path to a YOLO dataset folder (must contain ``data.yaml``).
    project_name:
        Name for the training project. Defaults to the dataset folder name.
    workspace_dir:
        Root workspace directory. Defaults to ``~/.pyzm/training``.
    model:
        Base YOLO model name (e.g. ``"yolo11s"``).
    epochs:
        Number of training epochs.
    batch:
        Batch size. ``None`` = auto-detect from hardware.
    imgsz:
        Training image size.
    val_ratio:
        Fraction of images used for validation.
    output:
        ONNX export path. ``None`` = auto in project dir.
    device:
        ``"auto"``, ``"cpu"``, or ``"cuda:0"`` etc.
    max_per_class:
        If > 0, keep at most this many images per class before importing.
    """
    dataset_path = Path(dataset_path).resolve()
    workspace = Path(workspace_dir) if workspace_dir else _DEFAULT_WORKSPACE

    # --- 1. Validate --------------------------------------------------------
    print(f"Validating dataset: {dataset_path}")
    result = validate_yolo_folder(dataset_path)
    if isinstance(result, str):
        raise ValueError(result)

    names_map: dict[int, str] = result["names"]
    splits = result["_splits"]
    class_names = [names_map[k] for k in sorted(names_map)]
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")

    # --- 2. Import ----------------------------------------------------------
    name = project_name or dataset_path.name
    project_dir = workspace / name
    project_dir.mkdir(parents=True, exist_ok=True)

    ds = YOLODataset(project_dir, classes=class_names)
    ds.init_project()
    store = VerificationStore(project_dir)

    print("Importing dataset...")
    img_count, det_count = import_local_dataset(
        ds, store, splits, names_map,
        max_per_class=max_per_class,
        progress_callback=_print_progress,
    )
    print(f"  {img_count} images, {det_count} annotations")

    # --- 3. Split + YAML ----------------------------------------------------
    print(f"Splitting dataset (val_ratio={val_ratio})...")
    ds.split(val_ratio)
    yaml_path = ds.generate_yaml()
    print(f"  Dataset YAML: {yaml_path}")

    # --- 4. Train ------------------------------------------------------------
    hw = YOLOTrainer.detect_hardware()
    effective_batch = batch if batch is not None else hw.suggested_batch
    effective_device = device if device != "auto" else hw.device

    print(f"Training: model={model}, epochs={epochs}, batch={effective_batch}, "
          f"imgsz={imgsz}, device={effective_device}")

    trainer = YOLOTrainer(model, project_dir, device=device)
    train_result = trainer.train(
        yaml_path,
        epochs=epochs,
        batch=effective_batch,
        imgsz=imgsz,
    )

    # --- 5. Export -----------------------------------------------------------
    if train_result.best_model:
        onnx_path = trainer.export_onnx(output)
        print(f"ONNX exported: {onnx_path}")
    else:
        onnx_path = None
        print("No best model found — skipping ONNX export.")

    # --- Summary -------------------------------------------------------------
    print("\n--- Training Summary ---")
    print(f"  Best epoch:   {train_result.best_epoch}/{train_result.total_epochs}")
    print(f"  mAP50:        {train_result.final_mAP50:.4f}")
    print(f"  mAP50-95:     {train_result.final_mAP50_95:.4f}")
    print(f"  Model size:   {train_result.model_size_mb:.1f} MB")
    print(f"  Duration:     {train_result.elapsed_seconds:.0f}s")
    if onnx_path:
        print(f"  ONNX path:    {onnx_path}")

    return train_result
