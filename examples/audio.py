#!/usr/bin/env python3
"""pyzm v2 -- detect bird species in an audio file using BirdNET.

Prerequisites:
    /opt/zoneminder/venv/bin/pip install birdnet-analyzer

Usage:
    python audio.py <audio_file>
    python audio.py /path/to/recording.wav
    python audio.py /path/to/event.mp4

Any format ffmpeg can read (WAV, MP3, MP4, etc.) is supported.

See also:
    Detection options, multi-model pipelines:
        https://pyzmv2.readthedocs.io/en/latest/guide/detection.html
"""

import sys

from pyzm import Detector

if len(sys.argv) < 2:
    audio_path = input("Enter audio file path: ")
else:
    audio_path = sys.argv[1]

# BirdNET via the ml_sequence dict format (same as objectconfig.yml)
ml_options = {
    "general": {
        "model_sequence": "audio",
    },
    "audio": {
        "general": {
            "pattern": ".*",
            "same_model_sequence_strategy": "first",
        },
        "sequence": [
            {
                "name": "BirdNET",
                "enabled": "yes",
                "audio_framework": "birdnet",
                "birdnet_min_conf": 0.5,
                # Set lat/lon for seasonal species filtering (or -1 to disable)
                "birdnet_lat": -1,
                "birdnet_lon": -1,
                "birdnet_sensitivity": 1.0,
                "birdnet_overlap": 0.0,
            },
        ],
    },
}

detector = Detector.from_dict(ml_options)

# detect_audio() runs BirdNET on the audio file directly
result = detector.detect_audio(audio_path)

if result.matched:
    print(f"Species detected: {result.summary}")
    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%}")
else:
    print("No bird species detected")
