#!/usr/bin/env python3
"""pyzm v2 -- detect objects in a local image file.

Usage:
    python image.py <image_path>
"""

import sys

import yaml

from pyzm import __version__ as pyzm_version
from pyzm import Detector

print(f"Using pyzm version: {pyzm_version}")

if len(sys.argv) < 2:
    stream = input("Enter filename to analyze: ")
else:
    stream = sys.argv[1]

with open("/etc/zm/secrets.yml") as f:
    conf = yaml.safe_load(f) or {}
secrets = conf.get("secrets", {})

# ML options (same dict format as objectconfig.yml ml_sequence)
ml_options = {
    "general": {
        "model_sequence": "object,face,alpr",
        "disable_locks": "no",
    },
    "object": {
        "general": {
            "pattern": ".*",
            "same_model_sequence_strategy": "most",
        },
        "sequence": [
            {
                "name": "TPU for object detection",
                "enabled": "yes",
                "object_weights": "/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                "object_labels": "/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names",
                "object_min_confidence": "0.3",
                "object_framework": "coral_edgetpu",
            },
            {
                "name": "YOLO11s object detection",
                "enabled": "yes",
                "object_weights": "/var/lib/zmeventnotification/models/ultralytics/yolo11s.onnx",
                "object_labels": "/var/lib/zmeventnotification/models/yolov4/coco.names",
                "object_min_confidence": "0.3",
                "object_framework": "opencv",
                "object_processor": "cpu",
                "image_path": "/var/lib/zmeventnotification/images",
            },
        ],
    },
    "face": {
        "general": {"pattern": ".*", "same_model_sequence_strategy": "union"},
        "sequence": [
            {
                "name": "DLIB face recognition",
                "enabled": "yes",
                "face_detection_framework": "dlib",
                "known_images_path": "/var/lib/zmeventnotification/known_faces",
                "face_model": "cnn",
                "face_train_model": "cnn",
                "face_recog_dist_threshold": "0.6",
                "face_num_jitters": "1",
                "face_upsample_times": "1",
            },
        ],
    },
    "alpr": {
        "general": {
            "same_model_sequence_strategy": "first",
            "pre_existing_labels": ["car", "motorbike", "bus", "truck", "boat"],
        },
        "sequence": [
            {
                "alpr_service": "plate_recognizer",
                "alpr_key": secrets.get("PLATEREC_ALPR_KEY"),
                "platerec_min_dscore": "0.1",
                "platerec_min_score": "0.2",
            },
        ],
    },
}

# Detect on a local image file
detector = Detector.from_dict(ml_options)
result = detector.detect(stream)

print(f"LABELS: {result.labels}")
print(f"BOXES: {result.boxes}")
print(f"CONFIDENCES: {result.confidences}")
print(f"IMAGE DIMS: {result.image_dimensions}")
