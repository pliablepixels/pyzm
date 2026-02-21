"""Coral TPU face detection backend — merged from pyzm.ml.face_tpu.

Detection-only (no recognition): all faces labeled as configured
``unknown_face_name`` or "face". Fixes deprecated ``Image.ANTIALIAS``
→ ``Image.LANCZOS``.

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


class FaceTpuBackend(MLBackend, PortalockerMixin):
    """Coral TPU face detection-only backend.

    Uses pycoral adapters for TPU inference. Detection only — all faces
    are labeled as ``unknown_face_name`` (default "face").
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self.processor = "tpu"
        self._auto_lock = True
        self._init_lock()

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "face_tpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from pycoral.utils.edgetpu import make_interpreter

        logger.debug('|--------- Loading "%s" model from disk -------------|', self.name)
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

        if self._auto_lock:
            self.acquire_lock()

        try:
            logger.debug(
                "|---------- TPU face (input image: %dw*%dh) ----------|",
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
            face_min_confidence = self._config.min_confidence
            objs = detect.get_objects(self._model, face_min_confidence, scale)

            diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"

            if self._auto_lock:
                self.release_lock()
        except:
            if self._auto_lock:
                self.release_lock()
            raise

        logger.debug(
            "perf: processor:%s Coral TPU face detection took: %s",
            self.processor,
            diff_time,
        )

        unknown_face_name = self._config.options.get("unknown_face_name", "face")
        detections: list[Detection] = []

        for obj in objs:
            detections.append(
                Detection(
                    label=unknown_face_name,
                    confidence=float(obj.score),
                    bbox=BBox(
                        x1=int(round(obj.bbox.xmin)),
                        y1=int(round(obj.bbox.ymin)),
                        x2=int(round(obj.bbox.xmax)),
                        y2=int(round(obj.bbox.ymax)),
                    ),
                    model_name=self.name,
                    detection_type="face",
                )
            )

        logger.debug("Coral face detection only, skipping recognition phase")
        return detections
