"""BirdNET audio bird recognition backend.

Uses ``birdnet_analyzer`` (https://github.com/kahst/BirdNET-Analyzer) to
identify bird species from audio extracted from ZoneMinder events.

Tested against birdnet_analyzer v2.4.0.
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

# Dummy bounding box for audio detections (no spatial location).
_AUDIO_BBOX = BBox(x1=0, y1=0, x2=1, y2=1)


class BirdnetBackend(MLBackend):
    """BirdNET audio analysis backend.

    Unlike image backends, the real work happens in :meth:`detect_audio`
    rather than :meth:`detect`.  The pipeline calls ``detect_audio()``
    when ``ModelConfig.type == AUDIO``.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._loaded = False

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "BirdNET"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        try:
            from birdnet_analyzer import config as cfg
            from birdnet_analyzer import utils as birdnet_utils  # noqa: F401
        except ImportError:
            raise ImportError(
                "birdnet_analyzer is not installed. "
                "Install with: /opt/zoneminder/venv/bin/pip install birdnet-analyzer"
            )

        # Ensure the model file exists / is downloaded
        birdnet_utils.ensure_model_exists()

        # Set required config paths that birdnet_analyzer needs
        cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
        cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
        cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE
        cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH

        # Load labels into cfg.LABELS
        with open(cfg.LABELS_FILE) as f:
            cfg.LABELS = [line.strip() for line in f if line.strip()]

        self._loaded = True
        logger.info("%s: BirdNET model ready (%d labels)", self.name, len(cfg.LABELS))

    def detect(self, image: "np.ndarray") -> list[Detection]:
        """Audio backend does not process images -- always returns empty."""
        return []

    def detect_audio(
        self,
        audio_path: str,
        event_week: int = -1,
        monitor_lat: float = -1.0,
        monitor_lon: float = -1.0,
    ) -> list[Detection]:
        """Run BirdNET analysis on an audio file.

        Parameters
        ----------
        audio_path:
            Path to an audio file (WAV, MP4, etc. — anything ffmpeg can read).
        event_week:
            ISO week number (1-48) of the event, for seasonal filtering.
            -1 disables seasonal filtering.
        monitor_lat, monitor_lon:
            Latitude/longitude from the monitor DB record.  Used as
            fallback when config ``birdnet_lat``/``birdnet_lon`` are -1.

        Returns
        -------
        list[Detection]
            One Detection per species found above the confidence threshold,
            with the best confidence across all chunks.
        """
        if not self._loaded:
            self.load()

        import numpy as np
        from birdnet_analyzer import audio, config as cfg, model

        # Resolve lat/lon: config overrides monitor DB
        lat = self._config.birdnet_lat
        lon = self._config.birdnet_lon
        if lat == -1.0 and monitor_lat != -1.0:
            lat = monitor_lat
        if lon == -1.0 and monitor_lon != -1.0:
            lon = monitor_lon

        # Configure birdnet_analyzer globals
        cfg.LATITUDE = lat
        cfg.LONGITUDE = lon
        cfg.WEEK = event_week if event_week > 0 else -1
        cfg.MIN_CONFIDENCE = self._config.birdnet_min_conf
        cfg.SIGMOID_SENSITIVITY = self._config.birdnet_sensitivity
        cfg.SIG_OVERLAP = self._config.birdnet_overlap

        # Build species filter based on location/week
        if lat != -1.0 and lon != -1.0:
            cfg.SPECIES_LIST = model.predict_filter(lat, lon, cfg.WEEK)
        else:
            cfg.SPECIES_LIST = []

        # Load audio and split into 3-second chunks
        sig, rate = audio.open_audio_file(audio_path, sample_rate=cfg.SAMPLE_RATE)
        chunks = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

        if not chunks:
            logger.debug("%s: no audio chunks from %s", self.name, audio_path)
            return []

        # Run predictions on all chunks, collect best confidence per species
        min_conf = self._config.birdnet_min_conf
        best_per_species: dict[str, float] = {}

        for chunk in chunks:
            prediction = model.predict([chunk])
            if prediction is None:
                continue

            # prediction is a numpy array of raw logits, apply sigmoid
            scores = model.flat_sigmoid(prediction[0], sensitivity=-cfg.SIGMOID_SENSITIVITY)

            # Find indices above threshold
            above = np.where(scores >= min_conf)[0]
            for idx in above:
                label = cfg.LABELS[idx]
                confidence = float(scores[idx])
                # Label format: "Scientific name_Common name"
                parts = label.split("_", 1)
                common_name = parts[1] if len(parts) > 1 else parts[0]

                if common_name not in best_per_species or confidence > best_per_species[common_name]:
                    best_per_species[common_name] = confidence

        # Convert to Detection objects
        detections: list[Detection] = []
        for species, conf in sorted(best_per_species.items(), key=lambda x: -x[1]):
            detections.append(
                Detection(
                    label=species,
                    confidence=conf,
                    bbox=_AUDIO_BBOX,
                    model_name=self.name,
                    detection_type="audio",
                )
            )

        if detections:
            det_summary = ", ".join(f"{d.label}:{d.confidence:.0%}" for d in detections)
            logger.debug("%s: %d species [%s]", self.name, len(detections), det_summary)
        else:
            logger.debug("%s: no bird species detected", self.name)

        return detections
