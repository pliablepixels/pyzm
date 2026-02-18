#!/usr/bin/env python3
"""pyzm v2 -- detect objects via a remote pyzm.serve server.

Two modes are supported:

  Image mode (default):
    Client JPEG-encodes the image and uploads it to the server.

  URL mode:
    Client sends ZM frame URLs to the server, which fetches
    images directly from ZoneMinder.  Useful when the GPU box
    has direct network access to ZM.

Prerequisites:
    Start the server on the GPU box:
        python -m pyzm.serve --models yolo11s --port 5000

Usage:
    python remote.py <image_path> [gateway_url]

See also:
    Server setup, authentication, YAML config, API reference:
        https://pyzmv2.readthedocs.io/en/latest/guide/serve.html
    Quick-start guide:
        https://pyzmv2.readthedocs.io/en/latest/guide/quickstart.html
"""

import sys

from pyzm import Detector

gateway = "http://localhost:5000"

if len(sys.argv) < 2:
    image_path = input("Enter filename to analyze: ")
else:
    image_path = sys.argv[1]

if len(sys.argv) >= 3:
    gateway = sys.argv[2]

# URL mode is the default -- detect_event() sends frame URLs and the
# server fetches them directly from ZoneMinder.
# Use gateway_mode="image" if the server can't reach ZM directly.
detector = Detector(models=["yolo11s"], gateway=gateway)
# detector = Detector(models=["yolo11s"], gateway=gateway, gateway_mode="image")

# With authentication:
# detector = Detector(
#     models=["yolo11s"],
#     gateway=gateway,
#     gateway_username="admin",
#     gateway_password="secret",
# )

# detect() always uploads the image (single-image mode)
result = detector.detect(image_path)

print(f"SUMMARY: {result.summary}")
for det in result.detections:
    print(f"  {det.label}: {det.confidence:.0%}")

# detect_event() uses URL mode by default -- sends frame URLs,
# server fetches them directly from ZM:
#
# from pyzm import ZMClient, StreamConfig
#
# zm = ZMClient(api_url="https://zm.example.com/zm/api",
#               user="admin", password="secret")
#
# result = detector.detect_event(
#     zm, event_id=12345,
#     stream_config=StreamConfig(frame_set=["snapshot", "alarm"]),
# )
# print(result.summary)
