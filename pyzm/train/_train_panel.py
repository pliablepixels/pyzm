"""Phase 3: Train & Export -- fine-tune and export ONNX."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from pyzm.train.app import MIN_IMAGES_PER_CLASS, _section_header
from pyzm.train.dataset import YOLODataset
from pyzm.train.trainer import HardwareInfo, TrainProgress, TrainResult, YOLOTrainer
from pyzm.train.verification import VerificationStore

logger = logging.getLogger("pyzm.train")


# ===================================================================
# PHASE 3: Train & Export
# ===================================================================

def _phase_train(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    _section_header("&#x1F3AF;", "Train & Export")
    pdir = st.session_state.get("workspace_dir")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    classes = store.build_class_list()
    if not classes:
        st.info("No verified detections yet. Go to **Review Detections** first.")
        return

    # Readiness check — only corrected classes need the image threshold
    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
    if needs:
        names = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in needs
        )
        st.warning(f"Need more images for: {names}")
        if st.button("Import More Images", key="train_go_select"):
            st.session_state["active_phase"] = "select"
            st.rerun()

    images = ds.staged_images()
    if len(images) < 2:
        st.info(
            f"You have {len(images)} image(s). We recommend at least "
            f"10\u201320 images per object type for effective training."
        )
        return

    hw = YOLOTrainer.detect_hardware()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        epochs = st.number_input(
            "Epochs", min_value=1, max_value=300, value=50, step=10,
            help="How many times to review all images during training. 50 is usually good for fine-tuning.",
        )
    with col2:
        batch = st.number_input(
            "Batch", min_value=1, max_value=128, value=hw.suggested_batch,
            help="Images processed at once. Auto-detected for your hardware.",
        )
    with col3:
        imgsz = st.selectbox(
            "Image size", [416, 640], index=1,
            help="Resolution for training. 640 is standard.",
        )
    with col4:
        st.caption(hw.display)

    # Shared mutable dict — background thread mutates contents,
    # main thread reads.  Lives in session_state so it survives reruns.
    shared: dict = st.session_state.get("_train_shared", {})

    if not shared.get("active", False):
        all_ready = not needs
        st.info(
            f"This fine-tuned model will **only** detect the following "
            f"{len(classes)} object(s): **{', '.join(classes)}**. "
            f"It will not retain the base model's original classes. "
            f"To detect other objects, run the base model alongside this one."
        )
        if hw.device == "cpu":
            st.warning(
                "Training on CPU will be significantly slower than GPU. "
                "Consider using a machine with a CUDA-capable GPU for faster training."
            )
        # Training duration estimate
        hw_factor = 1 if hw.device != "cpu" else 8
        est_minutes = int((len(images) * epochs) / (batch * 60) * hw_factor)
        if est_minutes > 0:
            st.caption(f":material/timer: Estimated training time: ~{est_minutes} minutes (rough estimate)")
        if st.button(":material/rocket_launch: Start Training", type="primary", disabled=not all_ready):
            class_name_to_id = {c: i for i, c in enumerate(classes)}
            ds.set_classes(classes)
            yaml_path = ds.generate_yaml()

            trainer = YOLOTrainer(
                base_model=base_model,
                project_dir=Path(pdir),
                device=hw.device,
            )
            shared = {
                "active": True,
                "progress": TrainProgress(
                    total_epochs=epochs, message="Preparing dataset...",
                ),
                "result": None,
                "log": [],
            }
            import time as _time
            shared["start_time"] = _time.time()
            shared["image_count"] = len(images)
            st.session_state["_train_shared"] = shared
            st.session_state["trainer"] = trainer
            st.session_state["classes"] = classes

            def _run(_s: dict = shared) -> None:
                # ── Dataset preparation (runs in background) ──
                # Skip rewriting annotations when the dataset was
                # imported with the same classes and nothing was modified.
                import_classes = ds.get_setting("import_classes")
                need_rewrite = (
                    store.has_modifications()
                    or import_classes != classes
                )

                n = len(images)
                if need_rewrite:
                    for i, img_path in enumerate(images):
                        anns = store.finalized_annotations(
                            img_path.name, class_name_to_id,
                        )
                        ds.update_annotations(img_path.name, anns)
                        if n > 100 and (i + 1) % 50 == 0:
                            _s["progress"] = TrainProgress(
                                total_epochs=epochs,
                                message=f"Writing annotations {i + 1}/{n}",
                            )

                # Re-split only if annotations changed or no split exists yet
                split_map = ds.get_setting("split_map") or {}
                has_split = ds._train_images.exists() and any(ds._train_images.iterdir())
                if need_rewrite or not has_split:
                    _s["progress"] = TrainProgress(
                        total_epochs=epochs,
                        message="Splitting into train/val...",
                    )
                    ds.split()

                _s["progress"] = TrainProgress(
                    total_epochs=epochs, message="Loading model...",
                )

                # ── Training ──
                def _cb(p: TrainProgress) -> None:
                    _s["progress"] = p

                # Capture all output (logging + stdout/stderr)
                class _TrainLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        log = _s["log"]
                        log.append(self.format(record))
                        if len(log) > 200:
                            del log[:-200]

                class _StreamCapture:
                    """Tee writes to the original stream and the shared log."""
                    def __init__(self, original):
                        self._original = original
                    def write(self, text):
                        self._original.write(text)
                        if text.strip():
                            log = _s["log"]
                            log.append(text.rstrip())
                            if len(log) > 200:
                                del log[:-200]
                    def flush(self):
                        self._original.flush()
                    def __getattr__(self, name):
                        return getattr(self._original, name)

                handler = _TrainLogHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                ul_logger = logging.getLogger("ultralytics")
                ul_logger.addHandler(handler)
                ul_logger.setLevel(logging.INFO)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = _StreamCapture(old_stdout)
                sys.stderr = _StreamCapture(old_stderr)
                try:
                    r = trainer.train(
                        dataset_yaml=yaml_path, epochs=epochs,
                        batch=batch, imgsz=imgsz, progress_callback=_cb,
                    )
                    _s["result"] = r
                except Exception as exc:
                    _s["progress"] = TrainProgress(finished=True, error=str(exc))
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    ul_logger.removeHandler(handler)
                    _s["active"] = False

            threading.Thread(target=_run, daemon=True).start()
            st.rerun()
    else:
        st.warning("Do not close this browser tab while training is in progress. Training will be lost.")
        p: TrainProgress = shared.get("progress") or TrainProgress()
        if p.total_epochs > 0:
            pct = max(0.0, min(1.0, p.epoch / p.total_epochs))
            st.progress(pct, text=p.message or f"Epoch {p.epoch}/{p.total_epochs}")

        import time as _time
        start = shared.get("start_time")
        if start:
            elapsed = _time.time() - start
            mins, secs = divmod(int(elapsed), 60)
            hrs, mins = divmod(mins, 60)
            elapsed_str = f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins}:{secs:02d}"
        else:
            elapsed_str = "0:00"
        img_count = shared.get("image_count", 0)

        t1, t2 = st.columns(2)
        t1.markdown(f":material/timer: **Elapsed Time:** {elapsed_str}")
        t2.markdown(f":material/image: **Training Images:** {img_count}")

        # User-friendly metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")

        # Box accuracy trend
        prev_box = st.session_state.get("_prev_box_loss")
        if prev_box is not None and p.box_loss > 0:
            if p.box_loss < prev_box - 0.001:
                box_label = "Box accuracy: Improving"
                box_color = "green"
            elif p.box_loss > prev_box + 0.001:
                box_label = "Box accuracy: Declining"
                box_color = "red"
            else:
                box_label = "Box accuracy: Stable"
                box_color = "orange"
        else:
            box_label = "Box accuracy: Starting"
            box_color = "gray"
        if p.epoch > 0:
            st.session_state["_prev_box_loss"] = p.box_loss
        m2.markdown(f":{box_color}[{box_label}]")

        # Detection quality from mAP50
        mAP_pct = p.mAP50 * 100
        if mAP_pct >= 80:
            quality = "Excellent"
        elif mAP_pct >= 60:
            quality = "Good"
        elif mAP_pct >= 30:
            quality = "Fair"
        else:
            quality = "Poor"
        m3.metric("Detection quality", f"{mAP_pct:.0f}% ({quality})")

        with st.expander("Technical Details"):
            td1, td2, td3, td4 = st.columns(4)
            td1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")
            td2.metric("Box Loss", f"{p.box_loss:.4f}")
            td3.metric("Cls Loss", f"{p.cls_loss:.4f}")
            td4.metric("mAP50", f"{p.mAP50:.3f}")

        if st.button(":material/stop_circle: Stop Training"):
            trainer = st.session_state.get("trainer")
            if trainer:
                trainer.request_stop()

        # Live training log
        train_log = shared.get("log", [])
        if train_log:
            st.code("\n".join(train_log[-30:]), language=None)

        if p.error:
            st.error(p.error)
        elif not p.finished:
            import time
            time.sleep(2)
            st.rerun()

    # Results + Export
    result: TrainResult | None = shared.get("result")
    if result:
        st.markdown(
            '<div style="background:#1B5E20;padding:0.8rem 1rem;'
            'border-radius:0.5rem;margin:1rem 0;">'
            '<h2 style="color:white;text-align:center;margin:0;">'
            '&#x2705; Training Complete</h2></div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("mAP50", f"{result.final_mAP50:.3f}")
        r2.metric("mAP50-95", f"{result.final_mAP50_95:.3f}")
        r3.metric("Size", f"{result.model_size_mb:.1f} MB")
        r4.metric("Time", f"{result.elapsed_seconds / 60:.1f} min")

        best_ep_label = (
            f"Epoch {result.best_epoch}/{result.total_epochs}"
            if result.best_epoch > 0
            else f"{result.total_epochs} epochs"
        )
        st.caption(f":material/star: Best model from: **{best_ep_label}**")

        if result.per_class:
            import pandas as pd

            rows = []
            for cls_name, cm in sorted(result.per_class.items()):
                rows.append({
                    "Class": cls_name,
                    "Precision": f"{cm.precision:.3f}",
                    "Recall": f"{cm.recall:.3f}",
                    "AP@50": f"{cm.ap50:.3f}",
                    "AP@50-95": f"{cm.ap50_95:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if result.best_model:
            st.code(str(result.best_model), language=None)

        pdir = st.session_state.get("workspace_dir")
        if pdir:
            st.caption(f"Dataset: `{pdir}`")
            _training_analysis(result, Path(pdir) / "runs" / "train")

        _phase_export(args)


def _training_analysis(result: TrainResult, train_dir: Path) -> None:
    """Show interpretive guidance and diagnostic images after training."""
    _section_header("&#x1F4CA;", "Training Analysis")

    # -- (a) Interpretive guidance --
    mAP = result.final_mAP50
    if mAP >= 0.8:
        st.success(f"**Excellent** — mAP50 {mAP:.2f} indicates strong detection quality.")
    elif mAP >= 0.6:
        st.info(f"**Good** — mAP50 {mAP:.2f}. Model performs well; more data may push it higher.")
    elif mAP >= 0.3:
        st.warning(f"**Moderate** — mAP50 {mAP:.2f}. Consider adding more diverse training images.")
    else:
        st.warning(f"**Poor** — mAP50 {mAP:.2f}. The model needs significantly more training data.")

    # Per-class weak spots
    weak = [name for name, cm in result.per_class.items() if cm.ap50 < 0.5]
    if weak:
        st.info(
            f"**Weak classes** (AP50 < 0.5): {', '.join(sorted(weak))}. "
            "Consider adding more training images for these."
        )

    # Overfitting hint
    if (
        result.best_epoch > 0
        and result.total_epochs > 0
        and result.best_epoch < result.total_epochs * 0.5
    ):
        st.info(
            f"Best model was at epoch {result.best_epoch}/{result.total_epochs} "
            "(early in training). The model may have overfit — "
            "try fewer epochs or more training data."
        )

    # -- (b) Training curves --
    results_png = train_dir / "results.png"
    if results_png.exists():
        st.markdown("##### Training Curves")
        with st.expander("How to read these curves"):
            st.markdown(
                "**Loss curves** (box_loss, cls_loss, dfl_loss): should decrease "
                "and flatten. If training loss keeps dropping but validation loss "
                "rises, the model is overfitting — stop earlier or add more data.\n\n"
                "**Precision & Recall**: both should rise and stabilize near 1.0. "
                "Low precision = too many false detections; low recall = missing "
                "real objects.\n\n"
                "**mAP50 / mAP50-95**: the main quality scores — higher is better. "
                "A plateau means more epochs won't help; more diverse data will."
            )
        st.image(str(results_png), width="stretch")

    # -- (c) Confusion matrix --
    cm_norm = train_dir / "confusion_matrix_normalized.png"
    cm_plain = train_dir / "confusion_matrix.png"
    cm_path = cm_norm if cm_norm.exists() else cm_plain if cm_plain.exists() else None
    if cm_path:
        st.markdown("##### Which Objects Get Mixed Up")
        with st.expander("How to read the confusion matrix"):
            st.markdown(
                "Each row is a **true** class, each column is a **predicted** class. "
                "Bright diagonal = correct predictions. Off-diagonal cells show "
                "which classes get confused with each other.\n\n"
                "A **background** row/column means missed detections (false negatives) "
                "or phantom detections (false positives). If a class has a high "
                "background score, the model needs more examples of that class."
            )
        st.image(str(cm_path), width="stretch")

    # -- (d) Evaluation curves --
    f1_path = train_dir / "F1_curve.png"
    pr_path = train_dir / "PR_curve.png"
    if f1_path.exists() or pr_path.exists():
        with st.expander("Detection Balance"):
            st.markdown(
                "**F1 Curve**: shows the balance between precision and recall at "
                "each confidence threshold. The peak is the optimal threshold — "
                "a sharp, high peak (close to 1.0) is ideal.\n\n"
                "**PR Curve**: precision vs. recall trade-off. A curve that hugs "
                "the top-right corner is a strong model. Area under the curve "
                "(AUC) equals AP — higher is better."
            )
            c1, c2 = st.columns(2)
            if f1_path.exists():
                c1.image(str(f1_path), caption="F1 Curve", width="stretch")
            if pr_path.exists():
                c2.image(str(pr_path), caption="PR Curve", width="stretch")

    # -- (e) Validation samples --
    val_labels = train_dir / "val_batch0_labels.jpg"
    val_preds = train_dir / "val_batch0_pred.jpg"
    if val_labels.exists() or val_preds.exists():
        with st.expander("Validation Samples"):
            st.markdown(
                "**Ground Truth** shows the actual labels. **Predictions** shows "
                "what the model detected. Compare them — missed objects or wrong "
                "labels indicate classes that need more training data."
            )
            c1, c2 = st.columns(2)
            if val_labels.exists():
                c1.image(str(val_labels), caption="Ground Truth", width="stretch")
            if val_preds.exists():
                c2.image(str(val_preds), caption="Predictions", width="stretch")


def _phase_export(args: argparse.Namespace) -> None:
    _section_header("&#x1F4E6;", "Export Model")
    pdir = st.session_state.get("workspace_dir")
    classes = st.session_state.get("classes", [])
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    best_pt = Path(pdir) / "runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        return

    st.info(
        f"This fine-tuned model only detects: **{', '.join(classes)}**. "
        f"To also detect standard objects (person, car, etc.), add both models "
        f"to your `objectconfig.yml` and use `same_model_sequence_strategy: union` "
        f"to merge results:\n"
        f"```yaml\n"
        f"same_model_sequence_strategy: union\n"
        f"```\n"
        f"Then list your base model and this fine-tuned model as separate entries "
        f"under the `object` sequence."
    )

    suggested_name = f"{base_model}_finetune.onnx"
    export_path = st.text_input(
        "Export ONNX to",
        value=str(Path(args.base_path) / "custom_finetune" / suggested_name),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(":material/download: Export ONNX", type="primary"):
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            with st.spinner("Exporting..."):
                try:
                    onnx_path = trainer.export_onnx(output_path=Path(export_path))
                    st.success(f"Exported: `{onnx_path}`")
                    st.code(
                        f"models:\n"
                        f"  - name: {onnx_path.stem}\n"
                        f"    type: object\n"
                        f"    framework: opencv\n"
                        f"    weights: {onnx_path}\n"
                        f"    min_confidence: 0.3\n"
                        f"    pattern: \"({'|'.join(classes)})\"\n",
                        language="yaml",
                    )
                    st.caption(
                        "Add this to your `objectconfig.yml` (usually at `/etc/zm/`) "
                        "under the `models:` section."
                    )
                except Exception as exc:
                    st.error(str(exc))

    with col2:
        test_file = st.file_uploader("Test image", type=["jpg", "jpeg", "png"], key="test_img")
        if test_file:
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            pil_img = Image.open(test_file).convert("RGB")
            img_array = np.array(pil_img)
            try:
                dets = trainer.evaluate(img_array[..., ::-1])
                from PIL import ImageDraw
                draw_img = pil_img.copy()
                draw = ImageDraw.Draw(draw_img)
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                    draw.text((x1, max(0, y1 - 12)), f"{d['label']} {d['confidence']:.0%}", fill="lime")
                st.image(draw_img, width="stretch")
                if dets:
                    st.caption(", ".join(f"{d['label']} {d['confidence']:.0%}" for d in dets))
                else:
                    st.caption("No detections")
            except Exception as exc:
                st.error(str(exc))
