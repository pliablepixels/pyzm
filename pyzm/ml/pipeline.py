"""ModelPipeline orchestrates the model sequence for a single frame.

It groups models by type (object, face, alpr), runs them in sequence with
the configured match strategy and fallback, applies pre_existing_labels
checks, and runs zone/size/pattern filters.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.ml.filters import (
    filter_by_pattern,
    filter_by_size,
    filter_by_zone,
    load_past_detections,
    match_past_detections,
    save_past_detections,
)
from pyzm.models.config import (
    DetectorConfig,
    MatchStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
    TypeOverrides,
)
from pyzm.models.detection import BBox, Detection, DetectionResult

if TYPE_CHECKING:
    import numpy as np
    from pyzm.models.zm import Zone

logger = logging.getLogger("pyzm.ml")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _create_backend(model_config: ModelConfig) -> MLBackend:
    """Create the right backend based on the model framework."""
    fw = model_config.framework

    if fw in (ModelFramework.OPENCV, ModelFramework.HOG, ModelFramework.VIRELAI):
        from pyzm.ml.backends.yolo import create_yolo_backend
        return create_yolo_backend(model_config)

    if fw == ModelFramework.CORAL:
        from pyzm.ml.backends.coral import CoralBackend
        return CoralBackend(model_config)

    if fw == ModelFramework.FACE_DLIB:
        from pyzm.ml.backends.face_dlib import FaceDlibBackend
        return FaceDlibBackend(model_config)

    if fw == ModelFramework.FACE_TPU:
        from pyzm.ml.backends.face_tpu import FaceTpuBackend
        return FaceTpuBackend(model_config)

    if fw in (ModelFramework.PLATE_RECOGNIZER, ModelFramework.OPENALPR):
        from pyzm.ml.backends.alpr import AlprBackend
        return AlprBackend(model_config)

    if fw == ModelFramework.REKOGNITION:
        from pyzm.ml.backends.rekognition import RekognitionBackend
        return RekognitionBackend(model_config)

    if fw == ModelFramework.BIRDNET:
        from pyzm.ml.backends.birdnet import BirdnetBackend
        return BirdnetBackend(model_config)

    raise ValueError(f"Unknown model framework: {fw}")


# ---------------------------------------------------------------------------
# ModelPipeline
# ---------------------------------------------------------------------------

class ModelPipeline:
    """Runs the full detection pipeline for a single image frame.

    The pipeline:

    1. Groups enabled models by :class:`ModelType` (object, face, alpr).
    2. For each type (in config order), runs the model variants in sequence
       applying the configured :class:`MatchStrategy`.
    3. Applies ``pre_existing_labels`` gates between types.
    4. Runs zone, size, and pattern filters on the combined results.
    """

    def __init__(self, detector_config: DetectorConfig) -> None:
        self._config = detector_config
        self._backends: list[tuple[ModelConfig, MLBackend]] = []
        self._loaded = False

        # Audio context (set before run() when audio models are in the sequence)
        self._audio_path: str | None = None
        self._audio_week: int = -1
        self._monitor_lat: float = -1.0
        self._monitor_lon: float = -1.0

    def set_audio_context(
        self,
        audio_path: str | None,
        event_week: int = -1,
        monitor_lat: float = -1.0,
        monitor_lon: float = -1.0,
    ) -> None:
        """Set audio file context for BirdNET / audio backends."""
        self._audio_path = audio_path
        self._audio_week = event_week
        self._monitor_lat = monitor_lat
        self._monitor_lon = monitor_lon

    # -- public API -----------------------------------------------------------

    def load(self) -> None:
        """Pre-load all enabled backends."""
        if self._loaded:
            return
        for mc in self._config.models:
            if not mc.enabled:
                logger.debug("Skipping disabled model: %s", mc.name or mc.framework)
                continue
            try:
                backend = _create_backend(mc)
                backend.load()
                self._backends.append((mc, backend))
            except Exception:
                logger.exception("Error loading model %s", mc.name or mc.framework)
        self._loaded = True

    def prepare(self) -> None:
        """Create backend objects without loading weights (lazy mode).

        Each backend will load its own weights on first ``detect()`` call
        thanks to the ``if self._model is None: self.load()`` guard in
        every backend's ``detect()`` method.
        """
        if self._loaded:
            return
        for mc in self._config.models:
            if not mc.enabled:
                continue
            try:
                backend = _create_backend(mc)
                # Don't call backend.load() — weights load on first detect()
                self._backends.append((mc, backend))
            except Exception:
                logger.exception("Error creating backend for %s", mc.name or mc.framework)
        self._loaded = True

    def run(
        self,
        image: "np.ndarray",
        zones: list["Zone"] | None = None,
        original_shape: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """Run the full pipeline on *image* and return a :class:`DetectionResult`.

        Parameters
        ----------
        image:
            BGR numpy array (OpenCV format).
        zones:
            Optional list of :class:`~pyzm.models.zm.Zone` objects.  If
            ``None``, a full-image zone is used.
        original_shape:
            ``(height, width)`` of the image *before* resizing.  When the
            image was resized for detection, zone polygons (which are in
            original coordinates) need to be rescaled to match.  ``None``
            means no rescaling is needed.
        """
        if not self._loaded:
            self.load()

        import numpy as np  # lazy

        h, w = image.shape[:2]

        # Prepare zone dicts, rescaling polygon points if the image was resized
        zone_dicts: list[dict] = []
        if zones:
            for z in zones:
                points = z.points
                if original_shape and (original_shape[0] != h or original_shape[1] != w):
                    orig_h, orig_w = original_shape
                    xfactor = w / orig_w
                    yfactor = h / orig_h
                    points = [(int(x * xfactor), int(y * yfactor)) for x, y in points]
                zone_dicts.append({
                    "name": z.name, "points": points,
                    "pattern": z.pattern, "ignore_pattern": z.ignore_pattern,
                })

        # Group backends by model type, preserving order
        type_groups: dict[ModelType, list[tuple[ModelConfig, MLBackend]]] = defaultdict(list)
        for mc, backend in self._backends:
            type_groups[mc.type].append((mc, backend))

        all_detections: list[Detection] = []
        all_error_boxes: list[BBox] = []

        # Track which types have been seen (for ordering by config)
        seen_types: list[ModelType] = []
        for mc in self._config.models:
            if mc.type not in seen_types and mc.enabled:
                seen_types.append(mc.type)

        for mtype in seen_types:
            variants = type_groups.get(mtype, [])
            if not variants:
                continue

            # Pre-existing labels check: if any variant in this type has
            # pre_existing_labels, verify that at least one was already detected.
            type_pre_existing = []
            for mc, _ in variants:
                if mc.pre_existing_labels:
                    type_pre_existing = mc.pre_existing_labels
                    break

            if type_pre_existing:
                existing_labels = [d.label for d in all_detections]
                if not any(lbl in existing_labels for lbl in type_pre_existing):
                    logger.debug(
                        "Skipping %s: pre_existing_labels %s not found in %s",
                        mtype.value, type_pre_existing, existing_labels,
                    )
                    continue

            best_for_type = self._run_model_variants(
                variants, image, zone_dicts, (h, w), mtype,
            )
            all_detections.extend(best_for_type)

        # Apply global pattern filter
        all_detections = filter_by_pattern(all_detections, self._config.pattern)

        # Apply global max_detection_size filter
        all_detections = filter_by_size(all_detections, self._config.max_detection_size, (h, w))

        # Apply zone filtering (returns kept and error_boxes)
        all_detections, all_error_boxes = filter_by_zone(all_detections, zone_dicts, (h, w))

        # Apply past-detection deduplication (per-type with global fallback)
        all_detections = self._filter_past_per_type(all_detections)

        return DetectionResult(
            detections=all_detections,
            image=image,
            image_dimensions={
                "original": original_shape or (h, w),
                "resized": (h, w) if original_shape else None,
            },
            error_boxes=all_error_boxes,
        )

    # -- private helpers ------------------------------------------------------

    def _resolve_type_overrides(self, mtype: ModelType) -> TypeOverrides:
        """Return per-type overrides for *mtype*, falling back to globals."""
        return self._config.type_overrides.get(mtype, TypeOverrides())

    def _filter_past_per_type(self, all_detections: list[Detection]) -> list[Detection]:
        """Apply past-detection filtering per model-type with global fallback.

        Loads past data once, groups detections by ``detection_type``, applies
        per-type config, then saves all surviving + new detections once.
        """
        import os

        cfg = self._config

        # Quick check: is past-detection matching enabled for *any* type?
        any_enabled = cfg.match_past_detections
        if not any_enabled:
            for tov in cfg.type_overrides.values():
                if tov.match_past_detections is True:
                    any_enabled = True
                    break
        if not any_enabled or not all_detections:
            return all_detections

        past_file = os.path.join(cfg.image_path, "past_detections.pkl")
        saved_boxes, saved_labels = load_past_detections(past_file)

        # Group detections by detection_type
        by_type: dict[str, list[Detection]] = defaultdict(list)
        for det in all_detections:
            by_type[det.detection_type].append(det)

        kept: list[Detection] = []
        for dtype, dets in by_type.items():
            # Resolve ModelType enum (if possible) to look up overrides
            try:
                mtype = ModelType(dtype)
            except ValueError:
                mtype = None

            tov = self._resolve_type_overrides(mtype) if mtype else TypeOverrides()

            enabled = tov.match_past_detections if tov.match_past_detections is not None else cfg.match_past_detections
            if not enabled:
                kept.extend(dets)
                continue

            max_diff = tov.past_det_max_diff_area if tov.past_det_max_diff_area is not None else cfg.past_det_max_diff_area
            label_overrides = tov.past_det_max_diff_area_labels or cfg.past_det_max_diff_area_labels
            ignore = tov.ignore_past_detection_labels if tov.ignore_past_detection_labels is not None else cfg.ignore_past_detection_labels
            aliases = tov.aliases if tov.aliases is not None else cfg.aliases

            kept.extend(match_past_detections(
                dets, saved_boxes, saved_labels,
                max_diff_area=max_diff,
                label_area_overrides=label_overrides,
                ignore_labels=ignore,
                aliases=aliases,
            ))

        save_past_detections(past_file, all_detections)
        return kept

    def _run_model_variants(
        self,
        variants: list[tuple[ModelConfig, MLBackend]],
        image: "np.ndarray",
        zone_dicts: list[dict],
        image_shape: tuple[int, int],
        mtype: ModelType | None = None,
    ) -> list[Detection]:
        """Run model variants for a single type using match_strategy."""
        tov = self._resolve_type_overrides(mtype) if mtype else TypeOverrides()
        strategy = tov.match_strategy if tov.match_strategy is not None else self._config.match_strategy
        best: list[Detection] = []

        for mc, backend in variants:
            # Per-variant pre_existing_labels check
            if mc.pre_existing_labels and best:
                existing = [d.label for d in best]
                if not any(lbl in existing for lbl in mc.pre_existing_labels):
                    logger.debug(
                        "Skipping variant %s: pre_existing_labels not met",
                        backend.name,
                    )
                    continue

            try:
                if mc.type == ModelType.AUDIO:
                    if not self._audio_path:
                        logger.debug("Skipping %s: no audio available", backend.name)
                        continue
                    raw = backend.detect_audio(
                        self._audio_path, self._audio_week,
                        self._monitor_lat, self._monitor_lon,
                    )
                else:
                    raw = backend.detect(image)
                if raw:
                    det_summary = ", ".join(
                        f"{d.label}:{d.confidence:.0%}" for d in raw
                    )
                    logger.debug("%s: %d detections [%s]", backend.name, len(raw), det_summary)
                else:
                    logger.debug("%s: no detections", backend.name)
            except Exception:
                logger.exception("Error running %s", backend.name)
                continue

            # Apply per-model pattern
            raw = filter_by_pattern(raw, mc.pattern)

            # Apply per-model max_detection_size
            raw = filter_by_size(raw, mc.max_detection_size, image_shape)

            if not raw:
                continue

            if strategy == MatchStrategy.FIRST:
                return raw

            if strategy == MatchStrategy.MOST:
                if len(raw) > len(best):
                    best = raw

            elif strategy == MatchStrategy.MOST_UNIQUE:
                if len({d.label for d in raw}) > len({d.label for d in best}):
                    best = raw

            elif strategy == MatchStrategy.UNION:
                best.extend(raw)

        return best
