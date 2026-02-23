#!/usr/bin/env python3
"""Quick-test a trained pyzm model on an image with bounding-box output.

Usage:
    python test_model.py <image> <onnx_weights> [--labels classes.txt] [--confidence 0.3] [--out output.jpg]

Example after `pyzm train`:
    python test_model.py photo.jpg ~/.pyzm/training/my_project/best.onnx
"""

import argparse
import sys
from pathlib import Path

import cv2

from pyzm import Detector
from pyzm.models.config import DetectorConfig, ModelConfig, ModelFramework, Processor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a trained model on an image and save annotated output."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("weights", help="Path to the ONNX weights file")
    parser.add_argument(
        "--labels", default=None,
        help="Path to labels/classes text file (optional — extracted from ONNX metadata if omitted)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Minimum confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output image path (default: <image>_detections.jpg)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"Weights not found: {weights_path}")

    if args.labels and not Path(args.labels).exists():
        sys.exit(f"Labels file not found: {args.labels}")

    out_path = Path(args.out) if args.out else image_path.with_name(
        f"{image_path.stem}_detections{image_path.suffix}"
    )

    # Build detector with the trained model
    detector = Detector(config=DetectorConfig(models=[
        ModelConfig(
            name=weights_path.stem,
            framework=ModelFramework.OPENCV,
            processor=Processor.CPU,
            weights=str(weights_path),
            labels=args.labels,
            min_confidence=args.confidence,
        ),
    ]))

    result = detector.detect(str(image_path))

    if not result.matched:
        print("No detections found.")
        return

    print(f"Detections: {result.summary}")
    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%}  "
              f"[{det.bbox.x1},{det.bbox.y1} -> {det.bbox.x2},{det.bbox.y2}]")

    annotated = result.annotate()
    cv2.imwrite(str(out_path), annotated)
    print(f"\nSaved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
