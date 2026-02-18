"""pyzm -- ZoneMinder Python library.

v2: Typed configuration, clean APIs, proper result objects.
"""

__version__ = "2.1.1"
VERSION = __version__

from pyzm.client import ZMClient
from pyzm.models.config import (
    DetectorConfig,
    ModelConfig,
    StreamConfig,
    ZMClientConfig,
)
from pyzm.models.detection import BBox, Detection, DetectionResult

__all__ = [
    "ZMClient",
    "Detector",
    "ZMClientConfig",
    "DetectorConfig",
    "ModelConfig",
    "StreamConfig",
    "BBox",
    "Detection",
    "DetectionResult",
]


def __getattr__(name: str):
    if name == "Detector":
        try:
            from pyzm.ml.detector import Detector
        except ImportError:
            raise ImportError(
                "ML dependencies are not installed. "
                'Install them with: pip install "pyzm[ml]"'
            ) from None
        return Detector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
