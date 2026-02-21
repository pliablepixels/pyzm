"""Merged YOLO backend — absorbs pyzm.ml.yolo base logic.

Provides :class:`YoloBase` (shared blob creation, NMS, GPU setup, locking)
and a :func:`create_yolo_backend` factory that dispatches to
:class:`YoloOnnx` or :class:`YoloDarknet` based on weights extension.

Refs #23
"""

from __future__ import annotations

import logging
import re
import time as _time
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend, PortalockerMixin
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


def _cv2_version() -> tuple[int, int, int]:
    """Return ``(major, minor, patch)`` from ``cv2.__version__``."""
    import cv2

    parts = [re.sub(r"[^0-9]", "", p) or "0" for p in cv2.__version__.split(".")]
    return (
        int(parts[0]) if len(parts) > 0 else 0,
        int(parts[1]) if len(parts) > 1 else 0,
        int(parts[2]) if len(parts) > 2 else 0,
    )


class YoloBase(MLBackend, PortalockerMixin):
    """Shared base for Darknet and ONNX YOLO backends.

    Subclasses must implement:
      - ``_load_model()``
      - ``_forward_and_parse(blob, width, height, conf_threshold)``
            → ``(class_ids, confidences, boxes)``
    """

    _DEFAULT_DIM = 416

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.net = None
        self.classes: list[str] | None = None
        self.processor = config.processor.value
        self.model_height = config.model_height or self._DEFAULT_DIM
        self.model_width = config.model_width or self._DEFAULT_DIM
        self._init_lock()

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "yolo"

    @property
    def is_loaded(self) -> bool:
        return self.net is not None

    def load(self) -> None:
        logger.info(
            "%s: loading YOLO model (processor=%s, weights=%s)",
            self.name,
            self.processor,
            self._config.weights,
        )
        self._load_model()

        # Detect GPU→CPU fallback
        if self.processor != self._config.processor.value:
            logger.warning(
                "%s: requested processor=%s but fell back to %s",
                self.name,
                self._config.processor.value,
                self.processor,
            )
        else:
            logger.debug("%s: running on %s", self.name, self.processor)

    def detect(self, image: "np.ndarray") -> list[Detection]:
        import cv2
        import numpy as np

        if self.net is None:
            self.load()

        Height, Width = image.shape[:2]
        logger.debug(
            "%s: detect extracted image dimensions as: %dw x %dh",
            self.name,
            Width,
            Height,
        )

        if self._auto_lock:
            self.acquire_lock()

        try:
            blob = self._create_blob(image)

            nms_threshold = 0.4
            conf_threshold = 0.2
            if self._config.min_confidence < conf_threshold:
                conf_threshold = self._config.min_confidence

            _t0 = _time.perf_counter()
            try:
                class_ids, confidences, boxes = self._forward_and_parse(
                    blob, Width, Height, conf_threshold
                )
            except cv2.error as e:
                if self.processor == "gpu":
                    logger.error(
                        "%s: GPU inference failed: %s. Falling back to CPU.",
                        self.name,
                        e,
                    )
                    self.processor = "cpu"
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    class_ids, confidences, boxes = self._forward_and_parse(
                        blob, Width, Height, conf_threshold
                    )
                else:
                    raise

            diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
            logger.debug(
                "perf: processor:%s %s detection took: %s",
                self.processor,
                self.name,
                diff_time,
            )

            if self._auto_lock:
                self.release_lock()
        except:
            if self._auto_lock:
                self.release_lock()
            raise

        # NMS
        _t0 = _time.perf_counter()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug(
            "perf: processor:%s %s NMS filtering took: %s",
            self.processor,
            self.name,
            diff_time,
        )
        indices = np.array(indices).flatten()

        detections: list[Detection] = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            conf = confidences[i]
            label = str(self.classes[class_ids[i]])

            if conf < self._config.min_confidence:
                logger.debug(
                    "%s: dropping %s (%.2f < %.2f)",
                    self.name,
                    label,
                    conf,
                    self._config.min_confidence,
                )
                continue

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=BBox(
                        x1=int(round(x)),
                        y1=int(round(y)),
                        x2=int(round(x + w)),
                        y2=int(round(y + h)),
                    ),
                    model_name=self.name,
                    detection_type="object",
                )
            )
        return detections

    # -- internal helpers (shared) --------------------------------------------

    def populate_class_labels(self) -> None:
        labels_path = self._config.labels
        with open(labels_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def _setup_gpu(self, cv2_ver: tuple[int, int, int]) -> None:
        """Configure CUDA backend if processor is 'gpu' and OpenCV supports it."""
        import cv2

        if self.processor == "gpu":
            if cv2_ver < (4, 2, 0):
                logger.error(
                    "%s: OpenCV %s does not support CUDA for DNNs (need 4.2+)",
                    self.name,
                    cv2.__version__,
                )
                self.processor = "cpu"
        else:
            logger.debug("%s: using CPU for detection", self.name)

        if self.processor == "gpu":
            logger.debug("%s: setting CUDA backend for OpenCV", self.name)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as e:
                logger.error(
                    "%s: failed to set CUDA backend: %s. Falling back to CPU.",
                    self.name,
                    e,
                )
                self.processor = "cpu"
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _create_blob(self, image: "np.ndarray"):
        import cv2

        scale = 0.00392  # 1/255
        return cv2.dnn.blobFromImage(
            image, scale, (self.model_width, self.model_height), (0, 0, 0), True, crop=False
        )

    # -- abstract (subclass) --------------------------------------------------

    def _load_model(self) -> None:
        raise NotImplementedError

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        raise NotImplementedError


def create_yolo_backend(config: ModelConfig) -> YoloBase:
    """Factory: return :class:`YoloOnnx` or :class:`YoloDarknet` based on weights extension."""
    weights = config.weights or ""
    if weights.lower().endswith(".onnx"):
        from pyzm.ml.backends.yolo_onnx import YoloOnnx

        return YoloOnnx(config)
    else:
        from pyzm.ml.backends.yolo_darknet import YoloDarknet

        return YoloDarknet(config)
