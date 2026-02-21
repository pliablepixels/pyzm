"""AWS Rekognition object detection backend — merged from pyzm.ml.aws_rekognition.

Refs #23
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class RekognitionBackend(MLBackend):
    """AWS Rekognition object detection backend.

    Cloud API — no model loading needed, no locking needed.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "aws_rekognition"

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def load(self) -> None:
        import boto3

        boto3_kwargs: dict = {}
        if self._config.aws_region:
            boto3_kwargs["region_name"] = self._config.aws_region
        if self._config.aws_access_key_id:
            boto3_kwargs["aws_access_key_id"] = self._config.aws_access_key_id
        if self._config.aws_secret_access_key:
            boto3_kwargs["aws_secret_access_key"] = self._config.aws_secret_access_key

        self._min_confidence = self._config.min_confidence
        if self._min_confidence < 1:
            # Rekognition wants confidence as 0%–100%, not 0.0–1.0
            self._min_confidence *= 100

        self._client = boto3.client("rekognition", **boto3_kwargs)
        logger.info(
            "%s: AWS Rekognition initialized (region=%s, min confidence: %.0f%%)",
            self.name,
            self._config.aws_region,
            self._min_confidence,
        )

    def detect(self, image: "np.ndarray") -> list[Detection]:
        import cv2

        if self._client is None:
            self.load()

        height, width = image.shape[:2]
        logger.debug(
            "|---------- AWS Rekognition (image: %dx%d) ----------|",
            width,
            height,
        )

        is_success, _buff = cv2.imencode(".jpg", image)
        if not is_success:
            logger.warning("Unable to convert image to JPG")
            return []
        image_jpg = _buff.tobytes()

        response = self._client.detect_labels(
            Image={"Bytes": image_jpg},
            MinConfidence=self._min_confidence,
        )
        logger.debug("Detection response: %s", response)

        detections: list[Detection] = []
        for item in response["Labels"]:
            if "Instances" not in item:
                continue
            for instance in item["Instances"]:
                if "BoundingBox" not in instance or "Confidence" not in instance:
                    continue
                label = item["Name"].lower()
                conf = instance["Confidence"] / 100
                bb = instance["BoundingBox"]
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=BBox(
                            x1=round(width * bb["Left"]),
                            y1=round(height * bb["Top"]),
                            x2=round(width * (bb["Left"] + bb["Width"])),
                            y2=round(height * (bb["Top"] + bb["Height"])),
                        ),
                        model_name=self.name,
                        detection_type="object",
                    )
                )
        return detections
