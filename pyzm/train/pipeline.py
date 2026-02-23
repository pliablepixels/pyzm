"""Headless training pipeline for ``python -m pyzm.train <dataset>``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pyzm.train.dataset import YOLODataset
from pyzm.train.local_import import (
    import_correct_model,
    import_local_dataset,
    validate_yolo_folder,
)
from pyzm.train.trainer import TrainResult, YOLOTrainer, adaptive_finetune_params
from pyzm.train.verification import VerificationStore

logger = logging.getLogger("pyzm.train")

_DEFAULT_WORKSPACE = Path.home() / ".pyzm" / "training"


def _print_progress(current: int, total: int, **_kw) -> None:
    pct = current * 100 // total
    sys.stdout.write(f"\rImporting... {current}/{total} ({pct}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def _print_detect_progress(
    current: int, total: int, name: str = "", summary: str = "",
) -> None:
    pct = current * 100 // total
    if name and summary:
        sys.stdout.write(f"\r  [{current}/{total}] {name}: {summary}")
    else:
        sys.stdout.write(f"\rImporting & detecting... {current}/{total} ({pct}%)")
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
    mode: str = "new_class",
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
    mode:
        ``"new_class"`` or ``"refine"`` — controls augmentation strategy.
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

    # --- Adaptive hyperparameters -------------------------------------------
    adaptive = adaptive_finetune_params(img_count, mode=mode)
    print(f"Adaptive fine-tuning ({adaptive['tier']} dataset, {adaptive['mode']} mode, "
          f"{img_count} images): freeze={adaptive['freeze']}, lr0={adaptive['lr0']}, "
          f"mosaic={adaptive['mosaic']}, erasing={adaptive['erasing']}, "
          f"patience={adaptive['patience']}, cos_lr={adaptive['cos_lr']}")

    # --- 3. Split + YAML ----------------------------------------------------
    # Respect explicit CLI --val-ratio; otherwise use adaptive value
    effective_val_ratio = val_ratio if val_ratio != 0.2 else adaptive["val_ratio"]
    print(f"Splitting dataset (val_ratio={effective_val_ratio})...")
    ds.split(effective_val_ratio)
    yaml_path = ds.generate_yaml()
    print(f"  Dataset YAML: {yaml_path}")

    # --- 4. Train ------------------------------------------------------------
    hw = YOLOTrainer.detect_hardware()
    effective_batch = batch if batch is not None else hw.suggested_batch
    effective_device = device if device != "auto" else hw.device

    train_extra = {k: adaptive[k] for k in (
        "freeze", "lr0", "patience", "cos_lr",
        "mosaic", "erasing", "scale", "mixup", "close_mosaic",
    )}
    print(f"Training: model={model}, epochs={epochs}, batch={effective_batch}, "
          f"imgsz={imgsz}, device={effective_device}")

    trainer = YOLOTrainer(model, project_dir, device=device)
    train_result = trainer.train(
        yaml_path,
        epochs=epochs,
        batch=effective_batch,
        imgsz=imgsz,
        **train_extra,
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


def run_correct_pipeline(
    image_folder: Path,
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
    base_path: str = "/var/lib/zmeventnotification/models",
    processor: str = "gpu",
    min_confidence: float = 0.3,
) -> TrainResult:
    """Run the correct-and-retrain pipeline headlessly.

    Steps: scan → import + detect → auto-approve all → build class list
    → split → train → export.  Always uses ``mode="refine"``.

    Parameters
    ----------
    image_folder:
        Path to a folder of images (no ``data.yaml`` needed).
    project_name:
        Name for the training project. Defaults to the folder name.
    workspace_dir:
        Root workspace directory. Defaults to ``~/.pyzm/training``.
    model:
        Base YOLO model name.
    base_path:
        Model base path passed to :class:`pyzm.ml.detector.Detector`.
    processor:
        ``"gpu"`` or ``"cpu"``.
    """
    image_folder = Path(image_folder).resolve()
    workspace = Path(workspace_dir) if workspace_dir else _DEFAULT_WORKSPACE

    if not image_folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {image_folder}")

    # --- 1. Project setup ----------------------------------------------------
    name = project_name or image_folder.name
    project_dir = workspace / name
    project_dir.mkdir(parents=True, exist_ok=True)

    ds = YOLODataset(project_dir, classes=[])
    ds.init_project()
    store = VerificationStore(project_dir)

    # --- 2. Import + detect --------------------------------------------------
    print(f"Scanning images in: {image_folder}")
    img_count, det_count = import_correct_model(
        ds, store, image_folder,
        base_model=model,
        workspace_dir=str(project_dir),
        base_path=base_path,
        processor=processor,
        min_confidence=min_confidence,
        progress_callback=_print_detect_progress,
    )
    print(f"  {img_count} images, {det_count} detections")

    if img_count == 0:
        raise ValueError(f"No images found in {image_folder}")

    # --- 3. Auto-approve all detections --------------------------------------
    from pyzm.train.verification import DetectionStatus

    approved = 0
    for img_name in store.all_images():
        iv = store.get(img_name)
        if iv is None:
            continue
        for det in iv.detections:
            if det.status == DetectionStatus.PENDING:
                det.status = DetectionStatus.APPROVED
                approved += 1
        iv.fully_reviewed = True
        store.set(iv)
    store.save()
    print(f"  Auto-approved {approved} detections")

    # --- 4. Build class list + split -----------------------------------------
    classes = store.build_class_list()
    if not classes:
        raise ValueError("No detections found — cannot build class list")
    print(f"  Classes ({len(classes)}): {', '.join(classes)}")

    ds.set_classes(classes)

    mode = "refine"
    adaptive = adaptive_finetune_params(img_count, mode=mode)
    print(f"Adaptive fine-tuning ({adaptive['tier']} dataset, {mode} mode, "
          f"{img_count} images): freeze={adaptive['freeze']}, lr0={adaptive['lr0']}, "
          f"mosaic={adaptive['mosaic']}, erasing={adaptive['erasing']}")

    effective_val_ratio = val_ratio if val_ratio != 0.2 else adaptive["val_ratio"]
    print(f"Splitting dataset (val_ratio={effective_val_ratio})...")

    # Write finalized annotations
    class_name_to_id = {c: i for i, c in enumerate(classes)}
    for img_path in ds.staged_images():
        anns = store.finalized_annotations(img_path.name, class_name_to_id)
        ds.update_annotations(img_path.name, anns)

    ds.split(effective_val_ratio)
    yaml_path = ds.generate_yaml()
    print(f"  Dataset YAML: {yaml_path}")

    # --- 5. Train ------------------------------------------------------------
    hw = YOLOTrainer.detect_hardware()
    effective_batch = batch if batch is not None else hw.suggested_batch
    effective_device = device if device != "auto" else hw.device

    train_extra = {k: adaptive[k] for k in (
        "freeze", "lr0", "patience", "cos_lr",
        "mosaic", "erasing", "scale", "mixup", "close_mosaic",
    )}
    print(f"Training: model={model}, epochs={epochs}, batch={effective_batch}, "
          f"imgsz={imgsz}, device={effective_device}")

    trainer = YOLOTrainer(model, project_dir, device=device)
    train_result = trainer.train(
        yaml_path,
        epochs=epochs,
        batch=effective_batch,
        imgsz=imgsz,
        **train_extra,
    )

    # --- 6. Export ------------------------------------------------------------
    if train_result.best_model:
        onnx_path = trainer.export_onnx(output)
        print(f"ONNX exported: {onnx_path}")
    else:
        onnx_path = None
        print("No best model found — skipping ONNX export.")

    # --- Summary --------------------------------------------------------------
    print("\n--- Training Summary ---")
    print(f"  Best epoch:   {train_result.best_epoch}/{train_result.total_epochs}")
    print(f"  mAP50:        {train_result.final_mAP50:.4f}")
    print(f"  mAP50-95:     {train_result.final_mAP50_95:.4f}")
    print(f"  Model size:   {train_result.model_size_mb:.1f} MB")
    print(f"  Duration:     {train_result.elapsed_seconds:.0f}s")
    if onnx_path:
        print(f"  ONNX path:    {onnx_path}")

    return train_result
