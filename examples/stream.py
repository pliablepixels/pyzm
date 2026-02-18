#!/usr/bin/env python3
"""pyzm v2 -- detect objects in a ZoneMinder event stream.

Usage:
    python stream.py <event_id> [<monitor_id>]
"""

import sys

import yaml

from pyzm import Detector, ZMClient, StreamConfig

if len(sys.argv) < 2:
    eid = input("Enter event ID to analyze: ")
    mid = input("Enter monitor ID (for zones): ")
else:
    eid = sys.argv[1]
    mid = sys.argv[2] if len(sys.argv) > 2 else input("Enter monitor ID: ")

# Read connection details from secrets
with open("/etc/zm/secrets.yml") as f:
    conf = yaml.safe_load(f) or {}
secrets = conf.get("secrets", {})

zm = ZMClient(
    api_url=secrets.get("ZM_API_PORTAL"),
    portal_url=secrets.get("ZM_PORTAL"),
    user=secrets.get("ZM_USER"),
    password=secrets.get("ZM_PASSWORD"),
)

stream_cfg = StreamConfig(
    frame_set=["snapshot", "alarm"],
    resize=800,
)

# Get zones for the monitor
m = zm.monitor(int(mid)) if mid else None
zones = m.get_zones() if m else None

# Run detection
detector = Detector(models=["yolo11s"])
result = detector.detect_event(zm, int(eid), zones=zones, stream_config=stream_cfg)

if result.matched:
    print(f"FRAME: {result.frame_id}")
    print(f"SUMMARY: {result.summary}")
    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%}")
else:
    print("No detections")
