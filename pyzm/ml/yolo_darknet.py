import logging
import time as _time

import cv2
import numpy as np

from pyzm.ml.yolo import YoloBase, cv2_version

logger = logging.getLogger("pyzm")


class YoloDarknet(YoloBase):
    """Darknet/OpenCV DNN backend for YOLO (.weights/.cfg models)."""

    def __init__(self, options={}):
        super().__init__(options, default_dim=416)
        self.is_get_unconnected_api_list = False

    def load_model(self):
        logger.debug('|--------- Loading "{}" model from disk -------------|'.format(self.name))
        _t0 = _time.perf_counter()
        self.net = cv2.dnn.readNet(
            self.options.get('object_weights'),
            self.options.get('object_config'),
        )
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"

        cv2_ver = cv2_version()
        if cv2_ver >= (4, 5, 4):
            logger.debug('{}: OpenCV >= 4.5.4, fixing getUnconnectedOutLayers() API'.format(self.name))
            self.is_get_unconnected_api_list = True

        logger.debug(
            'perf: processor:{} {} initialization (loading {} model from disk) took: {}'
            .format(self.processor, self.name, self.options.get('object_weights'), diff_time))

        self._setup_gpu(cv2_ver)
        self.populate_class_labels()

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        if self.is_get_unconnected_api_list:
            output_layers = [
                layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
            ]
        else:
            output_layers = [
                layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
            ]
        return output_layers

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        ln = self.get_output_layers()
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
