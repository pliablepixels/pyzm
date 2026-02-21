"""dlib face recognition backend — merged from pyzm.ml.face_dlib.

Absorbs all logic from legacy face_dlib.py: face_recognition, KNN matching,
unknown face saving. Lazy imports dlib and face_recognition in load().
Calls face_train_dlib.FaceTrain() via _build_train_options() helper.

Refs #23
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import time as _time
import uuid
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend, PortalockerMixin
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class FaceDlibBackend(MLBackend, PortalockerMixin):
    """dlib-based face recognition backend.

    Detects faces using HOG or CNN models via the ``face_recognition`` library,
    then compares against a KNN-trained known face database.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._knn = None
        self.processor = config.processor.value
        self.face_model = config.face_model
        self.upsample_times = config.face_upsample_times
        self.num_jitters = config.face_num_jitters
        self._init_lock()

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "face_dlib"

    @property
    def is_loaded(self) -> bool:
        return self._knn is not None

    def load(self) -> None:
        import dlib
        import face_recognition  # noqa: F401 — needed at runtime

        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
            self.processor = "gpu"
        else:
            self.processor = "cpu"

        logger.info(
            "%s: loading face recognition model (processor=%s, model=%s)",
            self.name,
            self.processor,
            self.face_model,
        )

        known_path = self._config.known_faces_dir
        encoding_file = os.path.join(known_path, "faces.dat")

        # Clean up legacy pickle
        old_pickle = os.path.join(known_path, "faces.pickle")
        if os.path.isfile(old_pickle):
            try:
                logger.debug("removing old faces.pickle, we have moved to clustering")
                os.remove(old_pickle)
            except Exception as e:
                logger.error("Error deleting old pickle file: %s", e)

        # Train if no encoding file exists
        if not os.path.isfile(encoding_file):
            logger.debug("trained file not found, reading from images and doing training...")
            import pyzm.ml.face_train_dlib as train

            train.FaceTrain(options=self._build_train_options()).train()

        try:
            with open(encoding_file, "rb") as f:
                self._knn = pickle.load(f)
        except Exception as e:
            logger.error("Error loading KNN model: %s", e)

    def detect(self, image: "np.ndarray") -> list[Detection]:
        import cv2
        import face_recognition
        import imutils

        if self._knn is None and not self.is_loaded:
            self.load()

        Height, Width = image.shape[:2]
        logger.debug(
            "|---------- Dlib Face recognition (input image: %dw*%dh) ----------|",
            Width,
            Height,
        )

        # Downscale large images for performance
        downscaled = False
        upsize_xfactor = None
        upsize_yfactor = None
        max_size = Width  # default: no downscale
        old_image = None

        if Width > max_size:
            downscaled = True
            logger.debug("Scaling image down to max size: %d", max_size)
            old_image = image.copy()
            image = imutils.resize(image, width=max_size)
            newHeight, newWidth = image.shape[:2]
            upsize_xfactor = Width / newWidth
            upsize_yfactor = Height / newHeight

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._auto_lock:
            self.acquire_lock()

        _t0 = _time.perf_counter()
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.face_model,
            number_of_times_to_upsample=self.upsample_times,
        )
        logger.debug(
            "perf: processor:%s finding faces took %s",
            self.processor,
            f"{(_time.perf_counter() - _t0) * 1000:.2f} ms",
        )

        _t0 = _time.perf_counter()
        face_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters,
        )

        if self._auto_lock:
            self.release_lock()

        logger.debug(
            "perf: processor:%s computing face recognition distances took %s",
            self.processor,
            f"{(_time.perf_counter() - _t0) * 1000:.2f} ms",
        )

        if not face_encodings:
            return []

        # KNN matching
        _t0 = _time.perf_counter()
        if self._knn:
            closest_distances = self._knn.kneighbors(face_encodings, n_neighbors=1)
            logger.debug("Closest knn match indexes: %s", closest_distances)
            are_matches = [
                closest_distances[0][i][0] <= self._config.face_recog_dist_threshold
                for i in range(len(face_locations))
            ]
            prediction_labels = self._knn.predict(face_encodings)
        else:
            are_matches = [False] * len(face_locations)
            prediction_labels = [""] * len(face_locations)
            logger.debug("No faces to match, creating empty set")

        logger.debug(
            "perf: processor:%s matching recognized faces took %s",
            self.processor,
            f"{(_time.perf_counter() - _t0) * 1000:.2f} ms",
        )

        # Upscale face locations if image was downscaled
        if downscaled:
            logger.debug("Scaling image back up to %d", Width)
            image = old_image
            new_face_locations = []
            for loc in face_locations:
                a, b, c, d = loc
                a = round(a * upsize_yfactor)
                b = round(b * upsize_xfactor)
                c = round(c * upsize_yfactor)
                d = round(d * upsize_xfactor)
                new_face_locations.append((a, b, c, d))
            face_locations = new_face_locations

        # Build detections
        unknown_face_name = self._config.options.get("unknown_face_name", "unknown")
        detections: list[Detection] = []

        for pred, loc, rec in zip(prediction_labels, face_locations, are_matches):
            label = pred if rec else unknown_face_name

            # Save unknown faces if configured
            if not rec and self._config.save_unknown_faces:
                self._save_unknown_face(image, loc)

            detections.append(
                Detection(
                    label=label,
                    confidence=1.0,
                    bbox=BBox(x1=loc[3], y1=loc[0], x2=loc[1], y2=loc[2]),
                    model_name=self.name,
                    detection_type="face",
                )
            )

        return detections

    # -- internal helpers -----------------------------------------------------

    def _build_train_options(self) -> dict:
        """Build the dict-based options that face_train_dlib.FaceTrain expects."""
        return {
            "known_images_path": self._config.known_faces_dir,
            "face_train_model": self._config.face_train_model,
            "face_upsample_times": self._config.face_upsample_times,
            "face_num_jitters": self._config.face_num_jitters,
        }

    def _save_unknown_face(self, image: "np.ndarray", loc: tuple) -> None:
        """Save a cropped unknown face to disk."""
        import cv2

        h, w = image.shape[:2]
        leeway = self._config.save_unknown_faces_leeway_pixels
        x1 = max(loc[3] - leeway, 0)
        y1 = max(loc[0] - leeway, 0)
        x2 = min(loc[1] + leeway, w)
        y2 = min(loc[2] + leeway, h)
        crop_img = image[y1:y2, x1:x2]

        timestr = time.strftime("%b%d-%Hh%Mm%Ss-")
        unknown_dir = self._config.unknown_faces_dir or "/tmp"
        unf = os.path.join(unknown_dir, timestr + str(uuid.uuid4()) + ".jpg")
        logger.info(
            "Saving cropped unknown face at [%d,%d,%d,%d - leeway=%dpx] to %s",
            x1,
            y1,
            x2,
            y2,
            leeway,
            unf,
        )
        cv2.imwrite(unf, crop_img)
