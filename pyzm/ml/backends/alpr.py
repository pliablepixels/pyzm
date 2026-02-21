"""ALPR (license plate recognition) backend — merged from pyzm.ml.alpr.

Supports PlateRecognizer, OpenALPR cloud, and OpenALPR command-line.

Bug fixes from legacy code:
  - alpr.py:259 ``url`` → ``self.url``
  - alpr.py:315 ``filename`` → ``self.filename``
  - alpr.py:321 ``options.get(...)`` → ``self._config.options.get(...)``
  - Security: ``subprocess.check_output(cmd, shell=True)``
    → ``subprocess.run(cmd_list)`` with ``shlex.split()``

Refs #23
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import uuid
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class AlprBackend(MLBackend):
    """ALPR backend with internal dispatch to PlateRecognizer, OpenALPR cloud,
    or OpenALPR command-line.

    No locking needed — API-based or short-lived subprocess.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._service: _AlprService | None = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "alpr"

    @property
    def is_loaded(self) -> bool:
        return self._service is not None

    def load(self) -> None:
        svc = self._config.alpr_service
        logger.info(
            "%s: initializing ALPR backend (service=%s)",
            self.name,
            svc,
        )
        if svc == "plate_recognizer":
            self._service = _PlateRecognizer(self._config)
        elif svc == "open_alpr":
            self._service = _OpenAlpr(self._config)
        elif svc == "open_alpr_cmdline":
            self._service = _OpenAlprCmdLine(self._config)
        else:
            raise ValueError(f'ALPR service "{svc}" not known')

    def detect(self, image: "np.ndarray") -> list[Detection]:
        if self._service is None:
            self.load()

        return self._service.detect(image, self.name)


# ---------------------------------------------------------------------------
# Internal service classes
# ---------------------------------------------------------------------------


class _AlprService:
    """Base for ALPR service implementations."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.url = config.alpr_url
        self.apikey = config.alpr_key
        self.filename: str | None = None
        self.remove_temp = False

    def detect(self, image: "np.ndarray", model_name: str) -> list[Detection]:
        raise NotImplementedError

    def _prepare(self, image_or_path) -> None:
        """Prepare the image: save blob to temp PNG or use existing file."""
        import cv2
        import imutils

        if not isinstance(image_or_path, str):
            logger.debug("Supplied object is not a file, assuming blob and creating file")
            max_size = self._config.max_detection_size
            if max_size:
                try:
                    max_px = int(max_size.rstrip("%").rstrip("px"))
                    logger.debug("resizing image blob to %d", max_px)
                    image_or_path = imutils.resize(
                        image_or_path, width=min(max_px, image_or_path.shape[1])
                    )
                except (ValueError, AttributeError):
                    pass
            self.filename = "/tmp/" + str(uuid.uuid4()) + "-alpr.png"
            cv2.imwrite(self.filename, image_or_path)
            self.remove_temp = True
        else:
            logger.debug("supplied object is a file %s", image_or_path)
            self.filename = image_or_path
            self.remove_temp = False

    def _cleanup(self) -> None:
        if self.remove_temp and self.filename and os.path.isfile(self.filename):
            os.remove(self.filename)


class _PlateRecognizer(_AlprService):
    """PlateRecognizer cloud/local API."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if not self.url:
            self.url = "https://api.platerecognizer.com/v1"
        logger.debug("PlateRecognizer ALPR initialized with url: %s", self.url)

    def detect(self, image: "np.ndarray", model_name: str) -> list[Detection]:
        import requests

        self._prepare(image)

        opts = self._config.options
        if opts.get("platerec_stats") == "yes":
            logger.debug("Plate Recognizer API usage stats: %s", json.dumps(self._stats()))

        try:
            with open(self.filename, "rb") as fp:
                platerec_url = self.url
                if opts.get("alpr_api_type", "cloud") == "cloud":
                    platerec_url += "/plate-reader"

                platerec_payload: dict = {}
                if opts.get("platerec_regions"):
                    platerec_payload["regions"] = opts["platerec_regions"]
                if opts.get("platerec_payload"):
                    logger.debug("Found platerec_payload, overriding payload")
                    platerec_payload = opts["platerec_payload"]
                if opts.get("platerec_config"):
                    logger.debug("Found platerec_config, using it")
                    platerec_payload["config"] = json.dumps(opts["platerec_config"])

                response = requests.post(
                    platerec_url,
                    timeout=15,
                    files=dict(upload=fp),
                    data=platerec_payload,
                    headers={"Authorization": "Token " + (self.apikey or "")},
                )
                response.raise_for_status()
                response = response.json()
                logger.debug("ALPR JSON: %s", response)
        except Exception as e:
            response = {
                "error": f"Plate recognizer rejected the upload with: {e}",
                "results": [],
            }
            logger.error("Plate recognizer rejected the upload: %s", e)
        finally:
            self._cleanup()

        detections: list[Detection] = []
        for plate in response.get("results", []):
            label = plate["plate"]
            dscore = plate["dscore"]
            score = plate["score"]
            if dscore >= self._config.platerec_min_dscore and score >= self._config.platerec_min_score:
                detections.append(
                    Detection(
                        label=f"alpr:{label}",
                        confidence=score,
                        bbox=BBox(
                            x1=round(int(plate["box"]["xmin"])),
                            y1=round(int(plate["box"]["ymin"])),
                            x2=round(int(plate["box"]["xmax"])),
                            y2=round(int(plate["box"]["ymax"])),
                        ),
                        model_name=model_name,
                        detection_type="alpr",
                    )
                )
            else:
                logger.debug(
                    "ALPR: discarding plate:%s because dscore:%.2f/score:%.2f "
                    "not in range of configured dscore:%.2f score:%.2f",
                    label,
                    dscore,
                    score,
                    self._config.platerec_min_dscore,
                    self._config.platerec_min_score,
                )
        return detections

    def _stats(self) -> dict:
        import requests

        opts = self._config.options
        if opts.get("alpr_api_type") != "cloud":
            return {}
        try:
            headers = {"Authorization": "Token " + self.apikey} if self.apikey else {}
            r = requests.get(self.url + "/statistics/", headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}


class _OpenAlpr(_AlprService):
    """OpenALPR cloud API."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if not self.url:
            self.url = "https://api.openalpr.com/v2/recognize"
        logger.debug("Open ALPR initialized with url: %s", self.url)

    def detect(self, image: "np.ndarray", model_name: str) -> list[Detection]:
        import requests

        self._prepare(image)

        opts = self._config.options
        min_confidence = float(opts.get("openalpr_min_confidence", 0.3))

        try:
            with open(self.filename, "rb") as fp:
                params = ""
                if opts.get("openalpr_country"):
                    params += "&country=" + opts["openalpr_country"]
                if opts.get("openalpr_state"):
                    params += "&state=" + opts["openalpr_state"]
                if opts.get("openalpr_recognize_vehicle"):
                    params += "&recognize_vehicle=" + str(opts["openalpr_recognize_vehicle"])

                rurl = f"{self.url}?secret_key={self.apikey}{params}"
                logger.debug("Trying OpenALPR with url: %s", rurl)
                response = requests.post(rurl, files={"image": fp})
                response.raise_for_status()
                response = response.json()
                logger.debug("OpenALPR JSON: %s", response)
        except Exception as e:
            response = {
                "error": f"Open ALPR rejected the upload with {e}",
                "results": [],
            }
            logger.debug("Open ALPR rejected the upload with %s", e)
        finally:
            self._cleanup()

        detections: list[Detection] = []
        for plate in response.get("results", []):
            label = plate["plate"]
            conf = float(plate["confidence"]) / 100
            if conf < min_confidence:
                logger.debug(
                    "OpenALPR: discarding plate: %s (conf %.2f < min %.2f)",
                    label,
                    conf,
                    min_confidence,
                )
                continue

            if plate.get("vehicle"):
                veh = plate["vehicle"]
                for attribute in ["color", "make", "make_model", "year"]:
                    if veh.get(attribute):
                        label = label + "," + veh[attribute][0]["name"]

            coords = plate["coordinates"]
            detections.append(
                Detection(
                    label=f"alpr:{label}",
                    confidence=conf,
                    bbox=BBox(
                        x1=round(int(coords[0]["x"])),
                        y1=round(int(coords[0]["y"])),
                        x2=round(int(coords[2]["x"])),
                        y2=round(int(coords[2]["y"])),
                    ),
                    model_name=model_name,
                    detection_type="alpr",
                )
            )
        return detections


class _OpenAlprCmdLine(_AlprService):
    """OpenALPR command-line utility."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        opts = config.options
        binary = opts.get("openalpr_cmdline_binary", "alpr")
        params = opts.get("openalpr_cmdline_params", "")
        self._base_cmd = f"{binary} {params}"
        if self._base_cmd.lower().find("-j") == -1:
            logger.debug("Adding -j to OpenALPR for json output")
            self._base_cmd += " -j"

    def detect(self, image: "np.ndarray", model_name: str) -> list[Detection]:
        self._prepare(image)

        cmd = f"{self._base_cmd} {self.filename}"
        logger.debug("OpenALPR CmdLine executing: %s", cmd)

        # Security fix: use shlex.split instead of shell=True
        cmd_list = shlex.split(cmd)
        try:
            result = subprocess.run(
                cmd_list, capture_output=True, text=True, check=False
            )
            response_text = result.stdout
            logger.debug("OpenALPR CmdLine response: %s", response_text)
            response = json.loads(response_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Error parsing JSON from command line: %s", e)
            response = {}
        finally:
            self._cleanup()

        opts = self._config.options
        min_confidence = float(opts.get("openalpr_cmdline_min_confidence", 0.3))

        detections: list[Detection] = []
        for plate in response.get("results", []):
            label = plate["plate"]
            conf = float(plate["confidence"]) / 100
            if conf < min_confidence:
                logger.debug(
                    "OpenALPR cmd line: discarding plate: %s (conf %.2f < min %.2f)",
                    label,
                    conf,
                    min_confidence,
                )
                continue

            coords = plate["coordinates"]
            detections.append(
                Detection(
                    label=f"alpr:{label}",
                    confidence=conf,
                    bbox=BBox(
                        x1=round(int(coords[0]["x"])),
                        y1=round(int(coords[0]["y"])),
                        x2=round(int(coords[2]["x"])),
                        y2=round(int(coords[2]["y"])),
                    ),
                    model_name=model_name,
                    detection_type="alpr",
                )
            )
        return detections
