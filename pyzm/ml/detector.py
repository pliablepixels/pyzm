"""Top-level Detector -- public API for pyzm v2 ML detection.

Usage::

    from pyzm import Detector

    # Auto-discover all models in the default path
    det = Detector()

    # Pick specific models by name (resolved from base_path)
    det = Detector(models=["yolo11s", "yolo26s"])

    # Custom model directory
    det = Detector(models=["yolo11s"], base_path="/my/models")

    # Detect
    result = det.detect("/path/to/image.jpg")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pyzm.ml.pipeline import ModelPipeline
from pyzm.models.config import (
    DetectorConfig,
    FrameStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
    Processor,
)
from pyzm.models.detection import DetectionResult

if TYPE_CHECKING:
    import numpy as np
    from pyzm.models.config import StreamConfig
    from pyzm.models.zm import Zone

logger = logging.getLogger("pyzm.ml")

DEFAULT_BASE_MODEL_PATH = "/var/lib/zmeventnotification/models"

# ---------------------------------------------------------------------------
# Model file discovery
# ---------------------------------------------------------------------------

# Extensions we recognise as model weight files
_WEIGHT_EXTS = {".weights", ".onnx", ".tflite"}
# Extensions we recognise as label files
_LABEL_EXTS = {".names", ".txt", ".labels"}


def _find_file(directory: Path, ext: str) -> Path | None:
    """Return the first file in *directory* matching the extension."""
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix == ext:
            return f
    return None


def _find_labels(directory: Path) -> str | None:
    """Find a labels file in *directory*, preferring .names > .txt > .labels."""
    for ext in (".names", ".txt", ".labels"):
        p = _find_file(directory, ext)
        if p:
            return str(p)
    return None


def _model_config_from_file(
    weights_path: Path,
    directory: Path,
    processor: Processor = Processor.CPU,
) -> ModelConfig:
    """Build a ModelConfig from a discovered weights file."""
    suffix = weights_path.suffix.lower()
    name = weights_path.stem

    if suffix == ".onnx":
        return ModelConfig(
            name=name,
            type=ModelType.OBJECT,
            framework=ModelFramework.OPENCV,
            processor=processor,
            weights=str(weights_path),
        )
    elif suffix == ".tflite":
        return ModelConfig(
            name=name,
            type=ModelType.OBJECT,
            framework=ModelFramework.CORAL,
            processor=Processor.TPU,
            weights=str(weights_path),
            labels=_find_labels(directory),
        )
    else:
        # .weights (Darknet)
        cfg = _find_file(directory, ".cfg")
        return ModelConfig(
            name=name,
            type=ModelType.OBJECT,
            framework=ModelFramework.OPENCV,
            processor=processor,
            weights=str(weights_path),
            config=str(cfg) if cfg else None,
            labels=_find_labels(directory),
        )


def _discover_models(
    base_path: Path,
    processor: Processor = Processor.CPU,
) -> list[ModelConfig]:
    """Scan *base_path* for model files and return ModelConfigs.

    Walks one level of subdirectories looking for weight files
    (.weights, .onnx, .tflite).
    """
    if not base_path.is_dir():
        logger.warning("Model base path %s does not exist", base_path)
        return []

    models: list[ModelConfig] = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir():
            continue
        for f in sorted(entry.iterdir()):
            if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                mc = _model_config_from_file(f, entry, processor)
                logger.debug("Discovered model: %s (%s)", mc.name, f)
                models.append(mc)

    if not models:
        logger.warning("No models found in %s", base_path)
    return models


def _resolve_model_name(
    name: str,
    base_path: Path,
    processor: Processor = Processor.CPU,
) -> ModelConfig:
    """Resolve a model name string against a base directory.

    Search order:
    1. Directory named *name* containing model files
    2. Any weight file whose stem matches *name* in any subdirectory
    3. Fall back to a bare ModelConfig with just the name and processor
    """
    # 1. Direct directory match: e.g. "yolo11" -> base_path/yolo11/
    candidate_dir = base_path / name
    if candidate_dir.is_dir():
        for f in sorted(candidate_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                return _model_config_from_file(f, candidate_dir, processor)

    # 2. File stem match across all subdirs: e.g. "yolo26s" -> ultralytics/yolo26s.onnx
    if base_path.is_dir():
        for subdir in sorted(base_path.iterdir()):
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.iterdir()):
                if f.is_file() and f.stem == name and f.suffix.lower() in _WEIGHT_EXTS:
                    return _model_config_from_file(f, subdir, processor)

    # 3. Fallback: bare config with no paths — will fail at load time
    #    if the backend actually needs weight files.
    logger.warning("Model '%s' not found in %s, creating bare config (no paths)", name, base_path)
    return ModelConfig(name=name, processor=processor)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """Top-level detection API.

    Parameters
    ----------
    config:
        A fully specified :class:`DetectorConfig`.  Takes precedence over
        *models* and *base_path*.
    models:
        A convenience shorthand -- a list of model name strings
        (e.g. ``["yolo11s", "yolo26s"]``) or :class:`ModelConfig` objects.
        String names are resolved against *base_path* to find weight,
        config, and label files automatically.
    base_path:
        Directory containing model subdirectories.  Defaults to
        ``/var/lib/zmeventnotification/models``.  When *models* is
        ``None`` and *config* is ``None``, all models in this directory
        are auto-discovered.  When *models* contains name strings, they
        are resolved against this path.
    processor:
        Hardware target for auto-discovered/resolved models.  Accepts
        ``"cpu"``, ``"gpu"``, ``"tpu"`` or a :class:`Processor` enum.
        Ignored when *config* is provided or when *models* contains
        :class:`ModelConfig` objects (which carry their own processor).
    gateway:
        URL of a remote ``pyzm.serve`` server (e.g. ``http://gpu:5000``).
        When set, ``detect()`` sends images to the remote server instead
        of running inference locally.
    gateway_mode:
        ``"image"`` (default) sends JPEG-encoded frames to the server.
        ``"url"`` sends frame URLs so the server fetches images directly
        from ZoneMinder.  Only applies to ``detect_event()``; single-image
        ``detect()`` calls always use image mode.
    gateway_timeout:
        HTTP timeout in seconds for remote detection requests.
    gateway_username:
        Username for remote server authentication (optional).
    gateway_password:
        Password for remote server authentication (optional).
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        models: list[str | ModelConfig] | None = None,
        base_path: str | Path = DEFAULT_BASE_MODEL_PATH,
        processor: str | Processor = Processor.CPU,
        *,
        gateway: str | None = None,
        gateway_mode: str = "image",
        gateway_timeout: int = 60,
        gateway_username: str | None = None,
        gateway_password: str | None = None,
    ) -> None:
        bp = Path(base_path)
        proc = Processor(processor) if isinstance(processor, str) else processor

        if config is not None:
            self._config = config
        elif models is not None:
            model_configs: list[ModelConfig] = []
            for m in models:
                if isinstance(m, str):
                    model_configs.append(_resolve_model_name(m, bp, proc))
                else:
                    model_configs.append(m)
            self._config = DetectorConfig(models=model_configs)
        else:
            # Auto-discover all models from base_path
            discovered = _discover_models(bp, proc)
            self._config = DetectorConfig(models=discovered)

        self._pipeline: ModelPipeline | None = None

        # Remote gateway
        self._gateway = gateway.rstrip("/") if gateway else None
        self._gateway_mode = gateway_mode
        self._gateway_timeout = gateway_timeout
        self._gateway_username = gateway_username
        self._gateway_password = gateway_password
        self._gateway_token: str | None = None

    # -- private helpers ------------------------------------------------------

    def _ensure_pipeline(self, lazy: bool = False) -> ModelPipeline:
        if self._pipeline is None:
            self._pipeline = ModelPipeline(self._config)
            if lazy:
                self._pipeline.prepare()
            else:
                self._pipeline.load()
        return self._pipeline

    @staticmethod
    def _load_image(path: str) -> "np.ndarray":
        import cv2  # lazy
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    # -- remote gateway helpers -----------------------------------------------

    def _ensure_gateway_token(self) -> str | None:
        """Authenticate with the remote gateway and cache the token."""
        if self._gateway_token:
            return self._gateway_token
        if not self._gateway_username:
            return None

        import requests

        resp = requests.post(
            f"{self._gateway}/login",
            json={
                "username": self._gateway_username,
                "password": self._gateway_password or "",
            },
            timeout=self._gateway_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._gateway_token = data.get("access_token")
        return self._gateway_token

    def _remote_detect(
        self,
        image: "np.ndarray",
        zones: list["Zone"] | None = None,
    ) -> DetectionResult:
        """Send an image to the remote gateway for detection."""
        import cv2
        import requests

        _, jpeg = cv2.imencode(".jpg", image)
        files = {"file": ("image.jpg", jpeg.tobytes(), "image/jpeg")}
        form_data: dict[str, str] = {}
        if zones:
            import json
            form_data["zones"] = json.dumps([z.as_dict() for z in zones])

        headers: dict[str, str] = {}
        token = self._ensure_gateway_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        resp = requests.post(
            f"{self._gateway}/detect",
            files=files,
            data=form_data,
            headers=headers,
            timeout=self._gateway_timeout,
        )
        resp.raise_for_status()
        result = DetectionResult.from_dict(resp.json())
        result.image = image
        return result

    def _remote_detect_urls(
        self,
        frame_urls: list[dict[str, str]],
        zm_auth: str,
        zones: list["Zone"] | None = None,
        verify_ssl: bool = True,
    ) -> DetectionResult:
        """Send frame URLs to the remote gateway for server-side fetching."""
        import requests

        headers: dict[str, str] = {}
        token = self._ensure_gateway_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload: dict[str, object] = {
            "urls": frame_urls,
            "zm_auth": zm_auth,
            "verify_ssl": verify_ssl,
        }
        if zones:
            payload["zones"] = [z.as_dict() for z in zones]

        resp = requests.post(
            f"{self._gateway}/detect_urls",
            json=payload,
            headers=headers,
            timeout=self._gateway_timeout,
        )
        resp.raise_for_status()
        return DetectionResult.from_dict(resp.json())

    # -- public API -----------------------------------------------------------

    def detect(
        self,
        input: "str | np.ndarray | list[tuple[int | str, np.ndarray]]",
        zones: list["Zone"] | None = None,
    ) -> DetectionResult:
        """Run detection on one or more images.

        Parameters
        ----------
        input:
            - ``str``: path to an image file.
            - ``np.ndarray``: a single BGR image array.
            - ``list[tuple[frame_id, np.ndarray]]``: multiple frames.
              The best frame is chosen by ``frame_strategy``.
        zones:
            Optional detection zone polygons.

        Returns
        -------
        DetectionResult
        """
        import numpy as np  # lazy

        if self._gateway:
            # Remote mode: send to gateway
            if isinstance(input, str):
                image = self._load_image(input)
                result = self._remote_detect(image, zones)
                result.frame_id = "single"
                return result
            if isinstance(input, np.ndarray):
                result = self._remote_detect(input, zones)
                result.frame_id = "single"
                return result
            if isinstance(input, list):
                return self._detect_multi_frame_remote(input, zones)
            raise TypeError(f"Unsupported input type: {type(input)}")

        pipeline = self._ensure_pipeline()

        # Single image path
        if isinstance(input, str):
            image = self._load_image(input)
            result = pipeline.run(image, zones=zones)
            result.frame_id = "single"
            return result

        # Single numpy array
        if isinstance(input, np.ndarray):
            result = pipeline.run(input, zones=zones)
            result.frame_id = "single"
            return result

        # Multiple frames: list of (frame_id, image) tuples
        if isinstance(input, list):
            return self._detect_multi_frame(input, zones, pipeline)

        raise TypeError(f"Unsupported input type: {type(input)}")

    def detect_event(
        self,
        zm_client: object,
        event_id: int,
        zones: list["Zone"] | None = None,
        stream_config: "StreamConfig | None" = None,
    ) -> DetectionResult:
        """Extract frames from a ZM event and run detection.

        Parameters
        ----------
        zm_client:
            A :class:`pyzm.zm.ZMClient` (or compatible) that provides
            ``get_event_frames(event_id, stream_config)`` returning
            ``(frames, image_dimensions)`` where *frames* is a list of
            ``(frame_id, np.ndarray)`` tuples and *image_dimensions* is a
            dict with ``'original'`` and ``'resized'`` keys.
        event_id:
            ZoneMinder event ID.
        zones:
            Optional detection zones.
        stream_config:
            Controls frame extraction (which frames, resize, etc.).

        Returns
        -------
        DetectionResult
        """
        from pyzm.models.config import StreamConfig as SC  # lazy

        sc = stream_config or SC()

        # URL mode: send frame URLs to the server instead of fetching frames locally
        if self._gateway and self._gateway_mode == "url":
            api = getattr(zm_client, "api", None)
            if api is None:
                raise AttributeError("zm_client.api required for URL-mode gateway")

            portal_url = api.portal_url
            auth_str = api.auth.get_auth_string()
            verify_ssl = api.config.verify_ssl

            frame_ids = sc.frame_set if sc.frame_set else ["snapshot"]
            frame_urls = [
                {"frame_id": str(fid), "url": f"{portal_url}/index.php?view=image&eid={event_id}&fid={fid}"}
                for fid in frame_ids
            ]
            return self._remote_detect_urls(frame_urls, auth_str, zones, verify_ssl)

        # Expect zm_client to have a method that returns frames
        get_frames = getattr(zm_client, "get_event_frames", None)
        if get_frames is None:
            raise AttributeError(
                "zm_client must provide a get_event_frames(event_id, stream_config) method"
            )

        result = get_frames(event_id, sc)

        # Unpack (frames, image_dims) tuple
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            frames, image_dims = result
            original_shape = image_dims.get("original")
        else:
            frames = result
            original_shape = None

        if not frames:
            logger.warning("No frames extracted for event %d", event_id)
            return DetectionResult()

        if self._gateway:
            return self._detect_multi_frame_remote(frames, zones, original_shape=original_shape)

        pipeline = self._ensure_pipeline()

        # Extract audio if any enabled model needs it
        wav_path = None
        has_audio_model = any(
            mc.type == ModelType.AUDIO and mc.enabled
            for mc in self._config.models
        )
        if has_audio_model:
            wav_path, week, mon_lat, mon_lon = self._extract_event_audio(
                zm_client, event_id,
            )
            pipeline.set_audio_context(wav_path, week, mon_lat, mon_lon)

        try:
            return self._detect_multi_frame(frames, zones, pipeline, original_shape=original_shape)
        finally:
            if wav_path:
                import os
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    # -- class methods --------------------------------------------------------

    @classmethod
    def from_config(cls, path: str) -> "Detector":
        """Load a Detector from a YAML configuration file.

        The YAML is expected to have a top-level structure that can be parsed
        into a :class:`DetectorConfig`.
        """
        import yaml  # lazy

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raw = {}

        detector_config = DetectorConfig.model_validate(raw)
        return cls(config=detector_config)

    @classmethod
    def from_dict(cls, ml_options: dict) -> "Detector":
        """Build a Detector from an ``ml_sequence`` dict.

        This delegates to :meth:`DetectorConfig.from_dict` so existing
        YAML configurations work directly.  If ``ml_options["general"]``
        contains ``ml_gateway``, the detector is created in remote mode.
        """
        detector_config = DetectorConfig.from_dict(ml_options)
        general = ml_options.get("general", {})
        return cls(
            config=detector_config,
            gateway=general.get("ml_gateway"),
            gateway_mode=general.get("ml_gateway_mode", "image"),
            gateway_username=general.get("ml_user"),
            gateway_password=general.get("ml_password"),
            gateway_timeout=int(general.get("ml_timeout", 60)),
        )

    # -- audio extraction -----------------------------------------------------

    @staticmethod
    def _extract_event_audio(
        zm_client: object,
        event_id: int,
    ) -> tuple[str | None, int, float, float]:
        """Extract audio from an event's video file for BirdNET analysis.

        Returns ``(wav_path, week, monitor_lat, monitor_lon)`` or
        ``(None, -1, -1.0, -1.0)`` on failure.
        """
        import os
        import subprocess
        import tempfile
        from datetime import datetime

        # Query DB for event video file and monitor location
        try:
            from pyzm.zm.db import get_zm_db
        except ImportError:
            logger.debug("pyzm.zm.db not available, skipping audio extraction")
            return None, -1, -1.0, -1.0

        conn = get_zm_db()
        if conn is None:
            logger.debug("Could not connect to ZM database, skipping audio extraction")
            return None, -1, -1.0, -1.0

        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT E.DefaultVideo, E.StartDateTime, "
                "M.Latitude, M.Longitude "
                "FROM Events E JOIN Monitors M ON E.MonitorId = M.Id "
                "WHERE E.Id = %s",
                (event_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
        except Exception:
            logger.debug("Failed to query event %d for audio extraction", event_id, exc_info=True)
            return None, -1, -1.0, -1.0

        if not row or not row.get("DefaultVideo"):
            logger.debug("Event %d has no DefaultVideo", event_id)
            return None, -1, -1.0, -1.0

        # Build the video file path via zm_client.event_path()
        try:
            video_dir = zm_client.event_path(event_id)
        except Exception:
            logger.debug("Failed to get event path for %d", event_id, exc_info=True)
            return None, -1, -1.0, -1.0

        video_path = os.path.join(video_dir, row["DefaultVideo"])
        if not os.path.isfile(video_path):
            logger.debug("Video file not found: %s", video_path)
            return None, -1, -1.0, -1.0

        # Probe for audio stream
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            if "audio" not in probe.stdout:
                logger.debug("No audio stream in %s", video_path)
                return None, -1, -1.0, -1.0
        except Exception:
            logger.debug("ffprobe failed for %s", video_path, exc_info=True)
            return None, -1, -1.0, -1.0

        # Extract audio to temp WAV (48 kHz mono, PCM s16le)
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="zm_birdnet_")
        os.close(wav_fd)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "48000", "-ac", "1",
                    wav_path,
                ],
                capture_output=True, timeout=60, check=True,
            )
        except Exception:
            logger.debug("ffmpeg audio extraction failed for %s", video_path, exc_info=True)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            return None, -1, -1.0, -1.0

        # Compute week number (1-48, clamped for BirdNET)
        week = -1
        start_dt = row.get("StartDateTime")
        if start_dt:
            if isinstance(start_dt, str):
                try:
                    start_dt = datetime.fromisoformat(start_dt)
                except ValueError:
                    start_dt = None
            if start_dt is not None:
                week = min((start_dt.timetuple().tm_yday // 7) + 1, 48)

        monitor_lat = float(row.get("Latitude") or -1.0)
        monitor_lon = float(row.get("Longitude") or -1.0)

        logger.debug(
            "Extracted audio for event %d: %s (week=%d, lat=%.2f, lon=%.2f)",
            event_id, wav_path, week, monitor_lat, monitor_lon,
        )
        return wav_path, week, monitor_lat, monitor_lon

    # -- multi-frame logic ----------------------------------------------------

    def _detect_multi_frame(
        self,
        frames: list[tuple[int | str, "np.ndarray"]],
        zones: list["Zone"] | None,
        pipeline: ModelPipeline,
        original_shape: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """Run detection on multiple frames and pick the best result using
        ``frame_strategy``."""
        strategy = self._config.frame_strategy
        all_results: list[DetectionResult] = []

        # Acquire session-level locks for exclusive-hardware backends
        # so the lock is held across ALL frames, not per-frame.
        locked_backends = []
        for _mc, backend in pipeline._backends:
            if backend.needs_exclusive_lock:
                backend.acquire_lock()
                locked_backends.append(backend)

        try:
            for frame_id, image in frames:
                try:
                    result = pipeline.run(image, zones=zones, original_shape=original_shape)
                    result.frame_id = frame_id
                    all_results.append(result)
                except Exception:
                    logger.exception("Error detecting frame %s", frame_id)
                    continue

                # Short-circuit for 'first' / 'first_new' strategies
                if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW) and result.matched:
                    logger.debug("Frame strategy %r: returning frame %s", strategy.value, frame_id)
                    return result

            if not all_results:
                return DetectionResult()

            # Pick best according to strategy
            best = all_results[0]
            for result in all_results[1:]:
                if _is_better(result, best, strategy):
                    best = result

            return best
        finally:
            for backend in locked_backends:
                backend.release_lock()

    def _detect_multi_frame_remote(
        self,
        frames: list[tuple[int | str, "np.ndarray"]],
        zones: list["Zone"] | None,
        original_shape: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """Run remote detection on multiple frames and pick the best result."""
        strategy = self._config.frame_strategy
        all_results: list[DetectionResult] = []

        for frame_id, image in frames:
            try:
                result = self._remote_detect(image, zones)
                result.frame_id = frame_id
                if original_shape:
                    result.image_dimensions["original"] = original_shape
                all_results.append(result)
            except Exception:
                logger.exception("Error in remote detection for frame %s", frame_id)
                continue

            if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW) and result.matched:
                logger.debug("Frame strategy %r: returning frame %s", strategy.value, frame_id)
                return result

        if not all_results:
            return DetectionResult()

        best = all_results[0]
        for result in all_results[1:]:
            if _is_better(result, best, strategy):
                best = result

        return best


def _is_better(
    candidate: DetectionResult,
    current: DetectionResult,
    strategy: FrameStrategy,
) -> bool:
    """Return True if *candidate* is a better result than *current* under the
    given frame strategy."""
    if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW):
        # Already handled by short-circuit above; fallback to first match
        return candidate.matched and not current.matched

    if strategy == FrameStrategy.MOST:
        if len(candidate.detections) != len(current.detections):
            return len(candidate.detections) > len(current.detections)
        return sum(candidate.confidences) > sum(current.confidences)

    if strategy == FrameStrategy.MOST_UNIQUE:
        cand_unique = len(set(candidate.labels))
        curr_unique = len(set(current.labels))
        if cand_unique != curr_unique:
            return cand_unique > curr_unique
        return sum(candidate.confidences) > sum(current.confidences)

    if strategy == FrameStrategy.MOST_MODELS:
        candidate_models = {d.model_name for d in candidate.detections}
        current_models = {d.model_name for d in current.detections}
        if len(candidate_models) != len(current_models):
            return len(candidate_models) > len(current_models)
        if len(candidate.detections) != len(current.detections):
            return len(candidate.detections) > len(current.detections)
        return sum(candidate.confidences) > sum(current.confidences)

    return False
