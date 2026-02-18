#!/usr/bin/env python3
"""pyzm v2 -- detect objects in a local image file.

Usage:
    python image.py <image_path>

See also:
    Multi-model configs, zones, from_dict():
        https://pyzmv2.readthedocs.io/en/latest/guide/detection.html
    Quick-start guide:
        https://pyzmv2.readthedocs.io/en/latest/guide/quickstart.html
"""

import sys

from pyzm import Detector

if len(sys.argv) < 2:
    image_path = input("Enter filename to analyze: ")
else:
    image_path = sys.argv[1]

# Model names are resolved against base_path on disk
detector = Detector(models=["yolo11s"])
result = detector.detect(image_path)

if result.matched:
    print(f"SUMMARY: {result.summary}")
    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%} at ({det.bbox.x1},{det.bbox.y1})-({det.bbox.x2},{det.bbox.y2})")
else:
    print("No detections")
