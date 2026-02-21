"""Darknet/OpenCV DNN YOLO backend — merged from pyzm.ml.yolo_darknet.

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


class YoloDarknet(YoloBase):
    """Darknet/OpenCV DNN backend for legacy YOLO (.weights/.cfg models)."""

    _DEFAULT_DIM = 416

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        self._is_get_unconnected_api_list = False

    def _load_model(self) -> None:
        import cv2

        logger.debug('|--------- Loading "%s" model from disk -------------|', self.name)
        _t0 = _time.perf_counter()
        self.net = cv2.dnn.readNet(self._config.weights, self._config.config)
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"

        cv2_ver = _cv2_version()
        if cv2_ver >= (4, 5, 4):
            logger.debug(
                "%s: OpenCV >= 4.5.4, fixing getUnconnectedOutLayers() API",
                self.name,
            )
            self._is_get_unconnected_api_list = True

        logger.debug(
            "perf: processor:%s %s initialization (loading %s from disk) took: %s",
            self.processor,
            self.name,
            self._config.weights,
            diff_time,
        )
        self._setup_gpu(cv2_ver)
        self.populate_class_labels()

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        if self._is_get_unconnected_api_list:
            return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        import numpy as np

        ln = self._get_output_layers()
        self.net.setInput(blob)
        outs = self.net.forward(ln)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

        return class_ids, confidences, boxes
