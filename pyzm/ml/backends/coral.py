"""Coral EdgeTPU object detection backend — merged from pyzm.ml.coral_edgetpu.

Refs #23
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend, PortalockerMixin
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class CoralBackend(MLBackend, PortalockerMixin):
    """Coral EdgeTPU object detection backend using pycoral.

    The EdgeTPU device is single-use, so this backend declares
    ``needs_exclusive_lock = True``. The pipeline holds the lock
    across the entire multi-frame session.
    """

    _auto_lock = False  # pipeline handles session-level locking

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self.processor = "tpu"
        self.classes: dict[int, str] = {}
        self._init_lock()

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "coral_edgetpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def needs_exclusive_lock(self) -> bool:
        return not self._config.disable_locks

    def load(self) -> None:
        from pycoral.utils.edgetpu import make_interpreter

        self._populate_class_labels()

        logger.info(
            "%s: loading Coral EdgeTPU model (processor=tpu, weights=%s)",
            self.name,
            self._config.weights,
        )
        _t0 = _time.perf_counter()
        self._model = make_interpreter(self._config.weights)
        self._model.allocate_tensors()
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug(
            "perf: processor:%s TPU initialization (loading %s from disk) took: %s",
            self.processor,
            self._config.weights,
            diff_time,
        )

    def detect(self, image: "np.ndarray") -> list[Detection]:
        import cv2
        from PIL import Image
        from pycoral.adapters import common, detect

        if self._model is None:
            self.load()

        Height, Width = image.shape[:2]
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        logger.debug(
            "|---------- TPU (input image: %dw*%dh) ----------|",
            Width,
            Height,
        )
        _t0 = _time.perf_counter()
        _, scale = common.set_resized_input(
            self._model,
            img.size,
            lambda size: img.resize(size, Image.LANCZOS),
        )
        self._model.invoke()
        objs = detect.get_objects(self._model, self._config.min_confidence, scale)

        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug(
            "perf: processor:%s Coral TPU detection took: %s",
            self.processor,
            diff_time,
        )

        detections: list[Detection] = []
        for obj in objs:
            label = self.classes.get(obj.id, str(obj.id))
            conf = float(obj.score)
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
                        x1=int(round(obj.bbox.xmin)),
                        y1=int(round(obj.bbox.ymin)),
                        x2=int(round(obj.bbox.xmax)),
                        y2=int(round(obj.bbox.ymax)),
                    ),
                    model_name=self.name,
                    detection_type="object",
                )
            )
        return detections

    # -- internal helpers -----------------------------------------------------

    def _populate_class_labels(self) -> None:
        labels_path = self._config.labels
        if not labels_path:
            raise ValueError("No label file provided for Coral model")
        with open(labels_path) as fp:
            for row in fp:
                classID, label = row.strip().split(" ", maxsplit=1)
                self.classes[int(classID)] = label.strip()
