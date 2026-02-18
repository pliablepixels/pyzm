#!/usr/bin/env python3
"""pyzm v2 quick-start examples.

Each section is self-contained.  Comment/uncomment what you need.

See also:
    Full documentation: https://pyzmv2.readthedocs.io/en/latest/
    Quick-start guide:  https://pyzmv2.readthedocs.io/en/latest/guide/quickstart.html
    Detection options:  https://pyzmv2.readthedocs.io/en/latest/guide/detection.html
    Remote server:      https://pyzmv2.readthedocs.io/en/latest/guide/serve.html
"""

import pyzm
print(f"pyzm {pyzm.__version__}")


# ============================================================================
# 1. ZM API CLIENT (no ML needed)
# ============================================================================

from pyzm import ZMClient

zm = ZMClient(
    api_url="https://demo.zoneminder.com/zm/api",
    user="zmuser",
    password="zmpass",
    # verify_ssl=False,  # for self-signed certs
)

print(f"ZM {zm.zm_version}, API {zm.api_version}")

# -- Monitors --
for m in zm.monitors():
    print(f"  Monitor {m.id}: {m.name} ({m.function}) {m.width}x{m.height}")

# -- Events (last hour) --
events = zm.events(since="1 hour ago", limit=5)
for ev in events:
    print(f"  Event {ev.id}: {ev.cause} ({ev.length:.1f}s, {ev.alarm_frames} alarm frames)")

# -- Single event --
if events:
    ev = zm.event(events[0].id)
    print(f"  Event detail: {ev.name} notes={ev.notes!r}")

# -- Zones for a monitor --
if zm.monitors():
    m = zm.monitors()[0]
    zones = m.get_zones()
    for z in zones:
        print(f"  Zone: {z.name} ({len(z.points)} points)")


# ============================================================================
# 2. ML DETECTION (no ZM needed)
# ============================================================================

from pyzm import Detector

# Quick start -- model names are resolved against base_path on disk
detector = Detector(models=["yolo11s"])

# Or with explicit config:
#
# from pyzm import DetectorConfig, ModelConfig
# from pyzm.models.config import ModelFramework, ModelType, Processor
#
# detector = Detector(config=DetectorConfig(
#     models=[ModelConfig(
#         type=ModelType.OBJECT,
#         framework=ModelFramework.OPENCV,
#         processor=Processor.GPU,
#         weights="/path/to/yolo11s.onnx",
#         labels="/path/to/coco.names",
#         min_confidence=0.5,
#     )],
# ))

# Detect on a local image
result = detector.detect("/tmp/image.jpg")

if result.matched:
    print(f"Detections: {result.summary}")
    # e.g. "person:97% car:85%"

    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%} at ({det.bbox.x1},{det.bbox.y1})-({det.bbox.x2},{det.bbox.y2})")

    # Draw boxes and save
    annotated = result.annotate()
    import cv2
    cv2.imwrite("/tmp/detected.jpg", annotated)
else:
    print("No detections")


# ============================================================================
# 3. ZM + ML TOGETHER (detect on a ZM event)
# ============================================================================

from pyzm import ZMClient, Detector, StreamConfig

zm = ZMClient(api_url="https://zm.example.com/zm/api", user="admin", password="secret")
detector = Detector(models=["yolo11s"])

event_id = 12345
m = zm.monitor(1)
zones = m.get_zones()

result = detector.detect_event(
    zm,
    event_id,
    zones=zones,
    stream_config=StreamConfig(
        frame_set=["snapshot", "alarm", "1"],
        resize=800,
    ),
)

if result.matched:
    print(result.summary)
    result.annotate()  # draw boxes on the image

    # Update ZM event notes
    ev = zm.event(event_id)
    ev.update_notes(result.summary)


# ============================================================================
# 4. AUDIO DETECTION (BirdNET)
# ============================================================================
# Requires: /opt/zoneminder/venv/bin/pip install birdnet-analyzer

from pyzm import Detector

# Audio-only config -- only audio models in the sequence
audio_opts = {
    "general": {"model_sequence": "audio"},
    "audio": {
        "general": {"pattern": ".*", "same_model_sequence_strategy": "first"},
        "sequence": [
            {
                "name": "BirdNET",
                "enabled": "yes",
                "audio_framework": "birdnet",
                "birdnet_min_conf": 0.5,
                "birdnet_lat": -1,       # -1 = no location filtering
                "birdnet_lon": -1,
                "birdnet_sensitivity": 1.0,
                "birdnet_overlap": 0.0,
            },
        ],
    },
}

detector = Detector.from_dict(audio_opts)

# detect_audio() works on any format ffmpeg can read (WAV, MP3, MP4, etc.)
result = detector.detect_audio("/tmp/recording.wav")

if result.matched:
    print(f"Birds: {result.summary}")
    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%}")
else:
    print("No bird species detected")


# ============================================================================
# 5. LOAD FROM YAML CONFIG (ml_sequence dict)
# ============================================================================

# If you already have an ml_sequence dict from your YAML config:
ml_sequence = {
    "general": {
        "model_sequence": "object,face",
    },
    "object": {
        "general": {"pattern": "(person|car|dog)", "same_model_sequence_strategy": "first"},
        "sequence": [
            {
                "object_framework": "coral_edgetpu",
                "object_weights": "/path/to/model.tflite",
                "object_labels": "/path/to/labels.txt",
                "object_min_confidence": 0.3,
            },
            {
                "object_framework": "opencv",
                "object_weights": "/path/to/yolo11s.onnx",
                "object_labels": "/path/to/coco.names",
                "object_processor": "gpu",
                "object_min_confidence": 0.5,
            },
        ],
    },
    "face": {
        "general": {"same_model_sequence_strategy": "first"},
        "sequence": [
            {
                "face_detection_framework": "dlib",
                "known_images_path": "/var/lib/zmeventnotification/known_faces",
                "face_model": "cnn",
                "face_recog_dist_threshold": 0.6,
            },
        ],
    },
}

detector = Detector.from_dict(ml_sequence)
result = detector.detect("/tmp/image.jpg")


# ============================================================================
# 6. LOGGING
# ============================================================================

# Standalone (ML only, no ZoneMinder):
import logging
logging.basicConfig(level=logging.DEBUG)
# All pyzm internals log to the "pyzm" logger -- this is all you need.

# With ZoneMinder (reads zm.conf + DB Config table automatically):
from pyzm.log import setup_zm_logging
adapter = setup_zm_logging(name="myapp", override={"dump_console": True})
adapter.Info("Hello from pyzm")
adapter.Debug(3, "Detail")
