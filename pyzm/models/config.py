"""Typed configuration models for pyzm v2.

All configuration is expressed as Pydantic models with sensible defaults.
Typos in field names cause immediate validation errors instead of silent failures.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, SecretStr, model_validator

logger = logging.getLogger("pyzm")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelType(str, Enum):
    """Kind of ML model in the detection pipeline."""
    OBJECT = "object"
    FACE = "face"
    ALPR = "alpr"
    AUDIO = "audio"


class ModelFramework(str, Enum):
    """Concrete ML backend implementation."""
    OPENCV = "opencv"
    CORAL = "coral_edgetpu"
    FACE_DLIB = "face_dlib"
    FACE_TPU = "face_tpu"
    PLATE_RECOGNIZER = "plate_recognizer"
    OPENALPR = "openalpr"
    REKOGNITION = "aws_rekognition"
    VIRELAI = "virelai"
    HOG = "hog"
    BIRDNET = "birdnet"


class Processor(str, Enum):
    """Hardware target for inference."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class MatchStrategy(str, Enum):
    """How to pick results when multiple model variants run for the same type."""
    FIRST = "first"
    MOST = "most"
    MOST_UNIQUE = "most_unique"
    UNION = "union"


class FrameStrategy(str, Enum):
    """How to pick the *best* frame across all analysed frames."""
    FIRST = "first"
    FIRST_NEW = "first_new"
    MOST = "most"
    MOST_UNIQUE = "most_unique"
    MOST_MODELS = "most_models"


# ---------------------------------------------------------------------------
# ZM connection
# ---------------------------------------------------------------------------

class ZMClientConfig(BaseModel):
    """Connection settings for ZoneMinder."""
    api_url: str
    portal_url: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    basic_auth_user: str | None = None
    basic_auth_password: SecretStr | None = None
    verify_ssl: bool = True
    timeout: int = 30

    # Database credential overrides (None = use zm.conf / built-in defaults)
    db_user: str | None = None
    db_password: SecretStr | None = None
    db_host: str | None = None
    db_name: str | None = None
    conf_path: str | None = None

    @model_validator(mode="after")
    def _derive_portal_url(self) -> "ZMClientConfig":
        if self.portal_url is None and self.api_url.endswith("/api"):
            self.portal_url = self.api_url[: -len("/api")]
        return self


# ---------------------------------------------------------------------------
# ML model configuration
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for a single ML model/backend in the pipeline."""
    name: str | None = None
    enabled: bool = True

    type: ModelType = ModelType.OBJECT
    framework: ModelFramework = ModelFramework.OPENCV
    processor: Processor = Processor.CPU

    # Model file paths
    weights: str | None = None
    config: str | None = None
    labels: str | None = None

    # Detection thresholds
    min_confidence: float = 0.3
    pattern: str = ".*"
    max_detection_size: str | None = None

    # Model input dimensions (YOLO-specific).
    # None = let the backend pick its own default (416 for Darknet, 640 for ONNX).
    model_width: int | None = None
    model_height: int | None = None

    # Face-specific
    known_faces_dir: str | None = None
    unknown_faces_dir: str | None = None
    save_unknown_faces: bool = False
    save_unknown_faces_leeway_pixels: int = 0
    face_model: str = "cnn"
    face_train_model: str = "cnn"
    face_recog_dist_threshold: float = 0.6
    face_num_jitters: int = 1
    face_upsample_times: int = 1

    # ALPR-specific
    alpr_service: str | None = None
    alpr_key: str | None = None
    alpr_url: str | None = None
    platerec_min_dscore: float = 0.1
    platerec_min_score: float = 0.2

    # BirdNET audio recognition
    birdnet_lat: float = -1.0
    birdnet_lon: float = -1.0
    birdnet_min_conf: float = 0.25
    birdnet_sensitivity: float = 1.0
    birdnet_overlap: float = 0.0

    # AWS Rekognition
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # Lock management
    disable_locks: bool = False
    max_lock_wait: int = 120
    max_processes: int = 1

    # Pre-condition: only run if a previous model already matched one of these labels
    pre_existing_labels: list[str] = Field(default_factory=list)

    # Arbitrary extra options for forward-compat
    options: dict[str, Any] = Field(default_factory=dict)


class TypeOverrides(BaseModel):
    """Per-model-type overrides (e.g. under ``object.general``, ``face.general``)."""
    match_strategy: MatchStrategy | None = None
    max_detection_size: str | None = None
    match_past_detections: bool | None = None
    past_det_max_diff_area: str | None = None
    past_det_max_diff_area_labels: dict[str, str] = Field(default_factory=dict)
    ignore_past_detection_labels: list[str] | None = None
    aliases: list[list[str]] | None = None


# Keys that may appear in section_general but only have global meaning.
_GLOBAL_ONLY_KEYS = frozenset({"frame_strategy", "image_path"})


class DetectorConfig(BaseModel):
    """Top-level configuration for the :class:`pyzm.ml.Detector`."""
    models: list[ModelConfig] = Field(default_factory=list)
    match_strategy: MatchStrategy = MatchStrategy.MOST
    frame_strategy: FrameStrategy = FrameStrategy.MOST_MODELS

    # Global overrides (applied to all models unless overridden per-model)
    max_detection_size: str | None = None
    pattern: str = ".*"

    # Past-detection matching
    match_past_detections: bool = False
    past_det_max_diff_area: str = "5%"
    past_det_max_diff_area_labels: dict[str, str] = Field(default_factory=dict)
    ignore_past_detection_labels: list[str] = Field(default_factory=list)
    aliases: list[list[str]] = Field(default_factory=list)
    image_path: str = "/tmp"

    # Per-type overrides (populated by from_dict when section_general has overridable keys)
    type_overrides: dict[ModelType, TypeOverrides] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, ml_options: dict[str, Any]) -> "DetectorConfig":
        """Build a DetectorConfig from a nested ``ml_sequence`` dict.

        This lets YAML configs work directly with the v2 API.
        """
        general = ml_options.get("general", {})
        model_sequence = general.get("model_sequence", "object").split(",")

        models: list[ModelConfig] = []
        type_overrides: dict[ModelType, TypeOverrides] = {}

        for model_type_str in model_sequence:
            model_type_str = model_type_str.strip()
            try:
                mtype = ModelType(model_type_str)
            except ValueError:
                continue

            section = ml_options.get(model_type_str, {})
            section_general = section.get("general", {})
            sequences = section.get("sequence", [])

            # Warn if global-only keys appear in section_general
            for gk in _GLOBAL_ONLY_KEYS:
                if gk in section_general:
                    logger.warning(
                        "%s.general.%s has no per-type effect; "
                        "move it to the top-level general section",
                        model_type_str, gk,
                    )

            # Build TypeOverrides from section_general (only if overridable keys present)
            tov = _build_type_overrides(section_general)
            if tov is not None:
                type_overrides[mtype] = tov

            # Resolve max_detection_size: section_general → global_general
            section_max_size = section_general.get("max_detection_size")

            for seq_item in sequences:
                mc = _seq_item_to_model_config(
                    mtype, seq_item, section_general, general,
                    section_max_size=section_max_size,
                )
                models.append(mc)

        strategy = MatchStrategy(general.get("same_model_sequence_strategy", "first"))
        frame_strat = FrameStrategy(general.get("frame_strategy", "most_models"))

        # Per-label past-detection area overrides: car_past_det_max_diff_area, etc.
        label_area_overrides: dict[str, str] = {}
        for k, v in general.items():
            if k.endswith("_past_det_max_diff_area") and k != "past_det_max_diff_area":
                label = k.removesuffix("_past_det_max_diff_area")
                label_area_overrides[label] = str(v)

        return cls(
            models=models,
            match_strategy=strategy,
            frame_strategy=frame_strat,
            max_detection_size=general.get("max_detection_size"),
            pattern=general.get("pattern", ".*"),
            match_past_detections=general.get("match_past_detections") == "yes",
            past_det_max_diff_area=general.get("past_det_max_diff_area", "5%"),
            past_det_max_diff_area_labels=label_area_overrides,
            ignore_past_detection_labels=general.get("ignore_past_detection_labels", []),
            aliases=general.get("aliases", []),
            image_path=general.get("image_path", "/tmp"),
            type_overrides=type_overrides,
        )


def _build_type_overrides(section_general: dict[str, Any]) -> TypeOverrides | None:
    """Build a :class:`TypeOverrides` from keys found in *section_general*.

    Returns ``None`` if no overridable keys are present.
    """
    kwargs: dict[str, Any] = {}

    if "same_model_sequence_strategy" in section_general:
        kwargs["match_strategy"] = MatchStrategy(section_general["same_model_sequence_strategy"])
    if "max_detection_size" in section_general:
        kwargs["max_detection_size"] = str(section_general["max_detection_size"])
    if "match_past_detections" in section_general:
        kwargs["match_past_detections"] = section_general["match_past_detections"] == "yes"
    if "past_det_max_diff_area" in section_general:
        kwargs["past_det_max_diff_area"] = str(section_general["past_det_max_diff_area"])
    if "ignore_past_detection_labels" in section_general:
        kwargs["ignore_past_detection_labels"] = section_general["ignore_past_detection_labels"]
    if "aliases" in section_general:
        kwargs["aliases"] = section_general["aliases"]

    # Per-label past-detection area overrides: car_past_det_max_diff_area, etc.
    label_overrides: dict[str, str] = {}
    for k, v in section_general.items():
        if k.endswith("_past_det_max_diff_area") and k != "past_det_max_diff_area":
            label = k.removesuffix("_past_det_max_diff_area")
            label_overrides[label] = str(v)
    if label_overrides:
        kwargs["past_det_max_diff_area_labels"] = label_overrides

    if not kwargs:
        return None
    return TypeOverrides(**kwargs)


def _seq_item_to_model_config(
    mtype: ModelType,
    seq: dict[str, Any],
    section_general: dict[str, Any],
    global_general: dict[str, Any],
    *,
    section_max_size: str | None = None,
) -> ModelConfig:
    """Convert one entry in a ``sequence`` list to a :class:`ModelConfig`."""
    prefix_map = {
        ModelType.OBJECT: "object",
        ModelType.FACE: "face",
        ModelType.ALPR: "alpr",
        ModelType.AUDIO: "audio",
    }
    prefix = prefix_map[mtype]

    framework_key = f"{prefix}_framework"
    if prefix == "face":
        framework_key = "face_detection_framework"
    default_fw = "birdnet" if mtype == ModelType.AUDIO else "opencv"
    fw_raw = seq.get(framework_key, default_fw)
    try:
        fw = ModelFramework(fw_raw)
    except ValueError:
        # map short name "dlib" -> "face_dlib"
        fw = ModelFramework(f"face_{fw_raw}") if mtype == ModelType.FACE else ModelFramework(fw_raw)

    processor_raw = seq.get(f"{prefix}_processor", seq.get("object_processor", "cpu"))
    try:
        processor = Processor(processor_raw)
    except ValueError:
        processor = Processor.CPU

    pre_existing = (
        seq.get("pre_existing_labels")
        or section_general.get("pre_existing_labels")
        or []
    )

    return ModelConfig(
        name=seq.get("name"),
        enabled=seq.get("enabled", "yes") != "no",
        type=mtype,
        framework=fw,
        processor=processor,
        weights=seq.get(f"{prefix}_weights"),
        config=seq.get(f"{prefix}_config"),
        labels=seq.get(f"{prefix}_labels"),
        min_confidence=float(seq.get(f"{prefix}_min_confidence", seq.get("object_min_confidence", 0.3))),
        pattern=section_general.get("pattern", global_general.get("pattern", ".*")),
        max_detection_size=str(v) if (v := seq.get("max_detection_size") or seq.get("max_size") or section_max_size) is not None else None,
        model_width=int(seq["model_width"]) if "model_width" in seq else None,
        model_height=int(seq["model_height"]) if "model_height" in seq else None,
        known_faces_dir=seq.get("known_images_path"),
        unknown_faces_dir=seq.get("unknown_images_path"),
        save_unknown_faces=seq.get("save_unknown_faces", "no") == "yes",
        save_unknown_faces_leeway_pixels=int(seq.get("save_unknown_faces_leeway_pixels", 0)),
        face_model=seq.get("face_model", "cnn"),
        face_train_model=seq.get("face_train_model", "cnn"),
        face_recog_dist_threshold=float(seq.get("face_recog_dist_threshold", 0.6)),
        face_num_jitters=int(seq.get("face_num_jitters", 1)),
        face_upsample_times=int(seq.get("face_upsample_times", 1)),
        alpr_service=seq.get("alpr_service"),
        alpr_key=seq.get("alpr_key"),
        alpr_url=seq.get("alpr_url"),
        platerec_min_dscore=float(seq.get("platerec_min_dscore", 0.1)),
        platerec_min_score=float(seq.get("platerec_min_score", 0.2)),
        birdnet_lat=float(seq.get("birdnet_lat", -1.0)),
        birdnet_lon=float(seq.get("birdnet_lon", -1.0)),
        birdnet_min_conf=float(seq.get("birdnet_min_conf", seq.get("birdnet_min_confidence", 0.25))),
        birdnet_sensitivity=float(seq.get("birdnet_sensitivity", 1.0)),
        birdnet_overlap=float(seq.get("birdnet_overlap", 0.0)),
        aws_region=seq.get("aws_region", "us-east-1"),
        aws_access_key_id=seq.get("aws_access_key_id"),
        aws_secret_access_key=seq.get("aws_secret_access_key"),
        disable_locks=(seq.get("disable_locks") or global_general.get("disable_locks", "no")) == "yes",
        pre_existing_labels=pre_existing if isinstance(pre_existing, list) else [],
    )


# ---------------------------------------------------------------------------
# Stream / frame extraction
# ---------------------------------------------------------------------------

class StreamConfig(BaseModel):
    """Controls how frames are extracted from a ZM event or video file."""
    frame_set: list[str] = Field(default_factory=lambda: ["snapshot", "alarm", "1"])
    max_frames: int = 0
    resize: int | None = 800
    start_frame: int = 1
    frame_skip: int = 1
    download: bool = False
    download_dir: str = "/tmp"
    delay: int = 0
    delay_between_frames: int = 0
    delay_between_snapshots: int = 0
    contig_frames_before_error: int = 5
    max_attempts: int = 1
    sleep_between_attempts: int = 3
    disable_ssl_cert_check: bool = True
    save_frames: bool = False
    save_frames_dir: str = "/tmp"
    delete_after_analyze: bool = False
    convert_snapshot_to_fid: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamConfig":
        """Build a StreamConfig from a ``stream_sequence`` dict.

        Handles the string-valued conventions of the YAML config:

        * ``resize='no'`` → ``None``, ``resize='800'`` → ``800``
        * ``frame_set='snapshot,alarm,1'`` → ``['snapshot', 'alarm', '1']``
        * ``'yes'``/``'no'`` strings → ``bool``

        Keys not relevant to StreamConfig (like ``api``, ``polygons``,
        ``mid``, ``frame_strategy``) are silently ignored.
        """
        def _bool(val: Any, default: bool = False) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("yes", "true", "1")
            return default

        kwargs: dict[str, Any] = {}

        # resize: 'no' -> None, numeric string -> int
        if "resize" in d:
            raw = d["resize"]
            if raw is None or (isinstance(raw, str) and raw.lower() == "no"):
                kwargs["resize"] = None
            else:
                kwargs["resize"] = int(raw)

        # frame_set: comma-separated string -> list
        if "frame_set" in d:
            raw = d["frame_set"]
            if isinstance(raw, str):
                kwargs["frame_set"] = [s.strip() for s in raw.split(",") if s.strip()]
            elif isinstance(raw, list):
                kwargs["frame_set"] = [str(s) for s in raw]

        # Simple int fields
        for key in (
            "max_frames", "start_frame", "frame_skip", "delay",
            "delay_between_frames", "delay_between_snapshots",
            "contig_frames_before_error", "max_attempts",
            "sleep_between_attempts",
        ):
            if key in d:
                kwargs[key] = int(d[key])

        # Simple bool fields
        for key in (
            "download", "disable_ssl_cert_check", "save_frames",
            "delete_after_analyze", "convert_snapshot_to_fid",
        ):
            if key in d:
                kwargs[key] = _bool(d[key])

        # String fields
        for key in ("download_dir", "save_frames_dir"):
            if key in d:
                kwargs[key] = str(d[key])

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Serve configuration
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    """Configuration for the pyzm ML detection server (``pyzm.serve``)."""
    host: str = "0.0.0.0"
    port: int = 5000
    models: list[str] = Field(default_factory=lambda: ["yolo11s"])
    base_path: str = "/var/lib/zmeventnotification/models"
    processor: Processor = Processor.CPU
    detector_config: DetectorConfig | None = None
    auth_enabled: bool = False
    auth_username: str = "admin"
    auth_password: SecretStr = SecretStr("")
    token_expiry_seconds: int = 3600
    token_secret: str = "change-me"

    @model_validator(mode="after")
    def _validate_models_all(self) -> "ServerConfig":
        if "all" in self.models and len(self.models) > 1:
            raise ValueError(
                "'all' cannot be combined with other model names; "
                "use --models all by itself"
            )
        return self
