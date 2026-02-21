"""ONNX/Ultralytics YOLO backend — merged from pyzm.ml.yolo_onnx.

Refs #23
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING

from pyzm.ml.backends.yolo import YoloBase, _cv2_version

if TYPE_CHECKING:
    import numpy as np

    from pyzm.models.config import ModelConfig

logger = logging.getLogger("pyzm.ml")


class YoloOnnx(YoloBase):
    """ONNX/Ultralytics backend for YOLO (.onnx models)."""

    _DEFAULT_DIM = 640

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        # Override default dims for ONNX
        if config.model_width is None:
            self.model_width = self._DEFAULT_DIM
        if config.model_height is None:
            self.model_height = self._DEFAULT_DIM

        self.is_end2end = False
        self.is_native_e2e = False
        self.pre_nms_layer: str | None = None
        # Letterbox state (set per-detect call)
        self._lb_scale = 1.0
        self._lb_pad_w = 0
        self._lb_pad_h = 0

    # -- model loading --------------------------------------------------------

    def _load_model(self) -> None:
        import cv2

        logger.debug('|--------- Loading "%s" model from disk -------------|', self.name)
        _t0 = _time.perf_counter()
        weights = self._config.weights
        cv2_ver = _cv2_version()
        if cv2_ver < (4, 13, 0):
            logger.warning(
                "%s: OpenCV %s may not support all ONNX operators. "
                "OpenCV 4.13+ is recommended for ONNX YOLOv26 models.",
                self.name,
                cv2.__version__,
            )
        logger.debug("%s: ONNX model detected, using readNetFromONNX", self.name)
        self.net = cv2.dnn.readNetFromONNX(weights)
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug(
            "perf: processor:%s %s initialization (loading %s from disk) took: %s",
            self.processor,
            self.name,
            weights,
            diff_time,
        )
        self._setup_gpu(cv2_ver)
        self.populate_class_labels()

    def populate_class_labels(self) -> None:
        onnx_classes = self._load_onnx_metadata()
        labels_path = self._config.labels
        if not labels_path:
            if onnx_classes:
                logger.debug(
                    "%s: loaded %d class labels from ONNX metadata",
                    self.name,
                    len(onnx_classes),
                )
                self.classes = onnx_classes
                return
            raise ValueError(
                f"{self.name}: no labels provided and ONNX metadata extraction failed"
            )
        with open(labels_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def _load_onnx_metadata(self) -> list[str] | None:
        """Load metadata from ONNX model (labels, end2end flag, pre-NMS layer)."""
        import ast as _ast

        try:
            import onnx

            model = onnx.load(self._config.weights)
            meta = {prop.key: prop.value for prop in model.metadata_props}

            if meta.get("end2end", "").lower() == "true":
                self.is_end2end = True

                out_dims = model.graph.output[0].type.tensor_type.shape.dim
                out_shape = [d.dim_value for d in out_dims]
                if len(out_shape) == 3 and out_shape[2] == 6:
                    self.is_native_e2e = True
                    logger.debug(
                        "%s: Native end-to-end model (YOLO26-style) detected, output shape=%s",
                        self.name,
                        out_shape,
                    )

                for i, node in enumerate(model.graph.node):
                    if (
                        node.op_type == "Transpose"
                        and i + 1 < len(model.graph.node)
                        and model.graph.node[i + 1].op_type == "Split"
                    ):
                        self.pre_nms_layer = "onnx_node!" + node.name
                        break

                if self.is_native_e2e:
                    if self.pre_nms_layer:
                        logger.debug(
                            "%s: Pre-NMS fallback layer: %s",
                            self.name,
                            self.pre_nms_layer,
                        )
                elif self.pre_nms_layer:
                    logger.debug(
                        "%s: End2end ONNX detected, will read pre-NMS layer: %s",
                        self.name,
                        self.pre_nms_layer,
                    )
                else:
                    logger.error(
                        "%s: End2end ONNX detected but could not find pre-NMS layer",
                        self.name,
                    )
                    self.is_end2end = False

            if "names" in meta:
                names_dict = _ast.literal_eval(meta["names"])
                max_id = max(int(k) for k in names_dict.keys())
                classes = [""] * (max_id + 1)
                for k, v in names_dict.items():
                    classes[int(k)] = v
                return classes
        except Exception as e:
            logger.debug("%s: failed to load ONNX metadata: %s", self.name, e)
        return None

    # -- blob / letterbox -----------------------------------------------------

    def _letterbox(self, image: "np.ndarray") -> "np.ndarray":
        """Resize with aspect ratio preserved and pad to model dimensions."""
        import cv2
        import numpy as np

        h, w = image.shape[:2]
        target_w, target_h = self.model_width, self.model_height

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        self._lb_scale = scale
        self._lb_pad_w = pad_w
        self._lb_pad_h = pad_h

        return padded

    def _create_blob(self, image: "np.ndarray"):
        """Override: letterbox then normalize to [0,1] BGR->RGB blob."""
        import cv2

        letterboxed = self._letterbox(image)
        scale = 0.00392  # 1/255
        return cv2.dnn.blobFromImage(
            letterboxed, scale, (self.model_width, self.model_height), (0, 0, 0), True, crop=False
        )

    # -- forward / parse ------------------------------------------------------

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        import numpy as np

        if self.is_native_e2e:
            return self._parse_native_e2e(blob, conf_threshold)

        self.net.setInput(blob)
        if self.pre_nms_layer:
            outs = self.net.forward(self.pre_nms_layer)
        else:
            outs = self.net.forward()

        output = outs[0] if isinstance(outs, (list, tuple)) else outs
        if output.ndim == 3:
            output = output.squeeze(0)

        if output.shape[0] < output.shape[1]:
            predictions = output.T
        else:
            predictions = output

        logger.debug(
            "%s: ONNX output shape=%s, predictions=%s, end2end=%s",
            self.name,
            output.shape,
            predictions.shape,
            self.is_end2end,
        )

        class_scores = predictions[:, 4:]
        best_class_ids = np.argmax(class_scores, axis=1)
        best_confidences = class_scores[np.arange(len(best_class_ids)), best_class_ids]

        mask = best_confidences >= conf_threshold
        filtered = predictions[mask]
        filtered_ids = best_class_ids[mask]
        filtered_confs = best_confidences[mask]

        if len(filtered) == 0:
            return [], [], []

        s = self._lb_scale
        pw = self._lb_pad_w
        ph = self._lb_pad_h

        if self.is_end2end:
            x1 = (filtered[:, 0] - pw) / s
            y1 = (filtered[:, 1] - ph) / s
            x2 = (filtered[:, 2] - pw) / s
            y2 = (filtered[:, 3] - ph) / s
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
        else:
            cx = filtered[:, 0]
            cy = filtered[:, 1]
            bw = filtered[:, 2]
            bh = filtered[:, 3]
            x = (cx - bw / 2 - pw) / s
            y = (cy - bh / 2 - ph) / s
            w = bw / s
            h = bh / s

        class_ids = filtered_ids.tolist()
        confidences = filtered_confs.tolist()
        boxes = np.stack([x, y, w, h], axis=1).tolist()

        return class_ids, confidences, boxes

    def _parse_native_e2e(self, blob, conf_threshold):
        """Parse native end-to-end output (YOLO26-style).

        Output shape is (N, 6) with columns [x1, y1, x2, y2, conf, class_id].
        Falls back to pre-NMS layer if OpenCV produces garbled output.
        """
        import numpy as np

        self.net.setInput(blob)
        outs = self.net.forward()

        output = outs[0] if isinstance(outs, (list, tuple)) else outs
        if output.ndim == 3:
            output = output.squeeze(0)

        logger.debug("%s: Native e2e output shape=%s", self.name, output.shape)

        confs = output[:, 4]

        # Garbled case 1: all confidences near-zero
        if confs.max() < 0.01:
            logger.debug(
                "%s: Native e2e confidences near-zero (max=%.6f), "
                "falling back to pre-NMS layer",
                self.name,
                confs.max(),
            )
            if self.pre_nms_layer:
                self.is_native_e2e = False
                return self._forward_and_parse(blob, 0, 0, conf_threshold)
            logger.error("%s: No pre-NMS fallback available", self.name)
            return [], [], []

        # Garbled case 2: many detections with identical confidence
        nonzero = confs[confs > 0.001]
        if len(nonzero) > 10:
            unique_count = len(np.unique(np.round(nonzero, 4)))
            if unique_count < max(3, len(nonzero) * 0.1):
                logger.debug(
                    "%s: Native e2e output looks garbled (%d unique conf values "
                    "across %d detections), falling back to pre-NMS layer",
                    self.name,
                    unique_count,
                    len(nonzero),
                )
                if self.pre_nms_layer:
                    self.is_native_e2e = False
                    return self._forward_and_parse(blob, 0, 0, conf_threshold)
                logger.error("%s: No pre-NMS fallback available", self.name)
                return [], [], []

        mask = output[:, 4] >= conf_threshold
        filtered = output[mask]

        if len(filtered) == 0:
            return [], [], []

        s = self._lb_scale
        pw = self._lb_pad_w
        ph = self._lb_pad_h

        x1 = (filtered[:, 0] - pw) / s
        y1 = (filtered[:, 1] - ph) / s
        x2 = (filtered[:, 2] - pw) / s
        y2 = (filtered[:, 3] - ph) / s

        class_ids = filtered[:, 5].astype(int).tolist()
        confidences = filtered[:, 4].tolist()
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()

        logger.debug(
            "%s: Native e2e: %d detections above %s",
            self.name,
            len(class_ids),
            conf_threshold,
        )

        return class_ids, confidences, boxes
