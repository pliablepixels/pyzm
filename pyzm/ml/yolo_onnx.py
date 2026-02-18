import logging
import time as _time

import cv2
import numpy as np

from pyzm.ml.yolo import YoloBase, cv2_version

logger = logging.getLogger("pyzm")


class YoloOnnx(YoloBase):
    """ONNX/Ultralytics backend for YOLO (.onnx models)."""

    def __init__(self, options={}):
        super().__init__(options, default_dim=640)
        self.is_end2end = False
        self.is_native_e2e = False
        self.pre_nms_layer = None
        # Letterbox state (set per-detect call)
        self._lb_scale = 1.0
        self._lb_pad_w = 0
        self._lb_pad_h = 0

    def _load_onnx_metadata(self):
        """Load metadata from ONNX model (labels, end2end flag, pre-NMS layer)."""
        import ast as _ast
        try:
            import onnx
            model = onnx.load(self.options.get('object_weights'))
            meta = {prop.key: prop.value for prop in model.metadata_props}

            if meta.get('end2end', '').lower() == 'true':
                self.is_end2end = True

                # Check output shape: native e2e (YOLO26) outputs (1, N, 6)
                out_dims = model.graph.output[0].type.tensor_type.shape.dim
                out_shape = [d.dim_value for d in out_dims]
                if len(out_shape) == 3 and out_shape[2] == 6:
                    self.is_native_e2e = True
                    logger.debug('{}: Native end-to-end model (YOLO26-style) detected, '
                                'output shape={}'.format(self.name, out_shape))

                # Always find pre-NMS layer as fallback
                for i, node in enumerate(model.graph.node):
                    if (node.op_type == 'Transpose'
                            and i + 1 < len(model.graph.node)
                            and model.graph.node[i + 1].op_type == 'Split'):
                        self.pre_nms_layer = 'onnx_node!' + node.name
                        break

                if self.is_native_e2e:
                    if self.pre_nms_layer:
                        logger.debug('{}: Pre-NMS fallback layer: {}'.format(
                        self.name, self.pre_nms_layer))
                elif self.pre_nms_layer:
                    logger.debug('{}: End2end ONNX detected, will read pre-NMS layer: {}'.format(
                        self.name, self.pre_nms_layer))
                else:
                    logger.error('{}: End2end ONNX detected but could not find pre-NMS layer'.format(self.name))
                    self.is_end2end = False

            if 'names' in meta:
                names_dict = _ast.literal_eval(meta['names'])
                max_id = max(int(k) for k in names_dict.keys())
                classes = [''] * (max_id + 1)
                for k, v in names_dict.items():
                    classes[int(k)] = v
                return classes
        except Exception as e:
            logger.debug('{}: Failed to load ONNX metadata: {}'.format(self.name, e))
        return None

    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        # Always load metadata to detect end2end flag;
        # also use embedded labels when no object_labels file is provided
        onnx_classes = self._load_onnx_metadata()
        if not class_file_abs_path:
            if onnx_classes:
                logger.debug('{}: Loaded {} class labels from ONNX metadata'.format(self.name, len(onnx_classes)))
                self.classes = onnx_classes
                return
            raise ValueError('{}: No object_labels provided and ONNX metadata extraction failed'.format(self.name))
        f = open(class_file_abs_path, 'r')
        self.classes = [line.strip() for line in f.readlines()]
        f.close()

    def load_model(self):
        logger.debug('|--------- Loading "{}" model from disk -------------|'.format(self.name))
        _t0 = _time.perf_counter()
        weights = self.options.get('object_weights')
        cv2_ver = cv2_version()
        if cv2_ver < (4, 13, 0):
            logger.warning('{}: OpenCV {} may not support all ONNX operators (e.g. TopK). '
                           'OpenCV 4.13+ is recommended for ONNX YOLOv26 models.'.format(self.name, cv2.__version__))
        logger.debug('{}: ONNX model detected, using readNetFromONNX'.format(self.name))
        self.net = cv2.dnn.readNetFromONNX(weights)
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"

        logger.debug(
            'perf: processor:{} {} initialization (loading {} model from disk) took: {}'
            .format(self.processor, self.name, weights, diff_time))

        self._setup_gpu(cv2_ver)
        self.populate_class_labels()

    def _letterbox(self, image):
        """Resize image with aspect ratio preserved and pad to model dimensions.

        Stores scale and padding offsets in self._lb_scale, self._lb_pad_w,
        self._lb_pad_h for coordinate de-projection in _forward_and_parse.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.model_width, self.model_height

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size with gray (114, 114, 114) — Ultralytics default
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        self._lb_scale = scale
        self._lb_pad_w = pad_w
        self._lb_pad_h = pad_h

        return padded

    def _create_blob(self, image):
        """Override base: letterbox then normalize to [0,1] BGR->RGB blob."""
        letterboxed = self._letterbox(image)
        scale = 0.00392  # 1/255
        return cv2.dnn.blobFromImage(letterboxed,
                                     scale, (self.model_width, self.model_height), (0, 0, 0),
                                     True,
                                     crop=False)

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        if self.is_native_e2e:
            return self._parse_native_e2e(blob, conf_threshold)

        self.net.setInput(blob)
        if self.pre_nms_layer:
            outs = self.net.forward(self.pre_nms_layer)
        else:
            outs = self.net.forward()

        # Output formats:
        #   Standard (non-end2end): (1, 4+C, N) with [cx, cy, w, h, cls_scores...]
        #   End2end pre-NMS layer:  (1, N, 4+C) with [x1, y1, x2, y2, cls_scores...]
        #     (the end2end graph converts cxcywh -> xyxy before the NMS ops)
        output = outs[0] if isinstance(outs, (list, tuple)) else outs
        if output.ndim == 3:
            output = output.squeeze(0)

        # Ensure predictions are (num_predictions, 4+num_classes)
        if output.shape[0] < output.shape[1]:
            predictions = output.T
        else:
            predictions = output

        logger.debug('{}: ONNX output shape={}, predictions={}, end2end={}'.format(
            self.name, output.shape, predictions.shape, self.is_end2end))

        # Vectorized: extract best class and confidence per prediction
        class_scores = predictions[:, 4:]
        best_class_ids = np.argmax(class_scores, axis=1)
        best_confidences = class_scores[np.arange(len(best_class_ids)), best_class_ids]

        # Filter by confidence threshold
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
            # End2end pre-NMS: pred[0:4] = [x1, y1, x2, y2] in letterbox space
            x1 = (filtered[:, 0] - pw) / s
            y1 = (filtered[:, 1] - ph) / s
            x2 = (filtered[:, 2] - pw) / s
            y2 = (filtered[:, 3] - ph) / s
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
        else:
            # Standard: pred[0:4] = [cx, cy, w, h] in letterbox space
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
        No NMS is needed — the model produces non-overlapping predictions.
        Falls back to pre-NMS layer if OpenCV produces garbled output.
        """
        self.net.setInput(blob)
        outs = self.net.forward()

        output = outs[0] if isinstance(outs, (list, tuple)) else outs
        if output.ndim == 3:
            output = output.squeeze(0)

        logger.debug('{}: Native e2e output shape={}'.format(self.name, output.shape))

        # Validate: garbled OpenCV output from broken NMS operators.
        confs = output[:, 4]

        # Garbled case 1: all confidences near-zero (OpenCV can't run NMS ops correctly)
        if confs.max() < 0.01:
            logger.debug('{}: Native e2e confidences near-zero (max={:.6f}), '
                         'falling back to pre-NMS layer'.format(self.name, confs.max()))
            if self.pre_nms_layer:
                self.is_native_e2e = False
                return self._forward_and_parse(blob, 0, 0, conf_threshold)
            logger.error('{}: No pre-NMS fallback available'.format(self.name))
            return [], [], []

        # Garbled case 2: many detections with identical confidence (e.g. all 0.1274).
        nonzero = confs[confs > 0.001]
        if len(nonzero) > 10:
            unique_count = len(np.unique(np.round(nonzero, 4)))
            if unique_count < max(3, len(nonzero) * 0.1):
                logger.debug('{}: Native e2e output looks garbled ({} unique conf values '
                             'across {} detections), falling back to pre-NMS layer'.format(
                                 self.name, unique_count, len(nonzero)))
                if self.pre_nms_layer:
                    self.is_native_e2e = False
                    return self._forward_and_parse(blob, 0, 0, conf_threshold)
                logger.error('{}: No pre-NMS fallback available'.format(self.name))
                return [], [], []

        # Filter by confidence
        mask = output[:, 4] >= conf_threshold
        filtered = output[mask]

        if len(filtered) == 0:
            return [], [], []

        s = self._lb_scale
        pw = self._lb_pad_w
        ph = self._lb_pad_h

        # De-letterbox: coords are [x1, y1, x2, y2] in letterbox space
        x1 = (filtered[:, 0] - pw) / s
        y1 = (filtered[:, 1] - ph) / s
        x2 = (filtered[:, 2] - pw) / s
        y2 = (filtered[:, 3] - ph) / s

        class_ids = filtered[:, 5].astype(int).tolist()
        confidences = filtered[:, 4].tolist()
        # Return as [x, y, w, h] for compatibility with base detect() method
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()

        logger.debug('{}: Native e2e: {} detections above {}'.format(
            self.name, len(class_ids), conf_threshold))

        return class_ids, confidences, boxes
