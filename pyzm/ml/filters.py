"""Pure functions for filtering detections.

Every function takes and returns :class:`Detection` / :class:`BBox` objects
from :mod:`pyzm.models.detection`.  Heavy dependencies (Shapely, pickle) are
imported at function level so they remain optional.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    pass

logger = logging.getLogger("pyzm.ml")


# ---------------------------------------------------------------------------
# Zone filtering
# ---------------------------------------------------------------------------

def filter_by_zone(
    detections: list[Detection],
    zones: list[dict],
    image_shape: tuple[int, int],
) -> tuple[list[Detection], list[BBox]]:
    """Keep only detections whose bounding box intersects at least one zone.

    Parameters
    ----------
    detections:
        Raw detections from a backend.
    zones:
        Each zone is a dict-like with ``name`` (str), ``points``
        (list of (x, y) tuples), and optionally ``pattern`` (str | None).
        These can be :class:`pyzm.models.zm.Zone` objects turned into dicts
        via :meth:`as_dict`, or simple dicts.
    image_shape:
        ``(height, width)`` of the analysed image.  If *zones* is empty a
        full-image zone is synthesised.

    Returns
    -------
    kept:
        Detections that intersect at least one zone and whose label matches
        the zone's pattern (or the default ``.*``).
    error_boxes:
        Bounding boxes of detections that were filtered out.
    """
    # No zones = no filtering, pass everything through.
    if not zones:
        return detections, []

    from shapely.geometry import Polygon  # optional dependency

    h, w = image_shape

    kept: list[Detection] = []
    error_boxes: list[BBox] = []

    for det in detections:
        bbox_poly = Polygon(det.bbox.as_polygon_coords())
        matched = False

        for zone in zones:
            # Normalise: accept Zone objects or dicts with 'value'/'points' key
            zone_points = zone.get("points") or zone.get("value", [])
            zone_pattern = zone.get("pattern")
            zone_ignore = zone.get("ignore_pattern")
            zone_name = zone.get("name", "unnamed")

            zone_poly = Polygon(zone_points)
            if not bbox_poly.intersects(zone_poly):
                logger.debug(
                    "filter_by_zone: %s does NOT intersect zone %s",
                    det.label, zone_name,
                )
                continue

            # Check ignore_pattern first -- suppress matching labels in this zone
            if zone_ignore and re.match(zone_ignore, det.label):
                logger.debug(
                    "filter_by_zone: %s intersects zone %s and matches ignore_pattern %s, suppressing",
                    det.label, zone_name, zone_ignore,
                )
                continue  # try other zones

            # Zone intersects -- now check the pattern
            pattern = zone_pattern or ".*"
            if re.match(pattern, det.label):
                logger.debug(
                    "filter_by_zone: %s intersects zone %s and matches pattern %s",
                    det.label, zone_name, pattern,
                )
                kept.append(det)
                matched = True
                break  # matched on first zone is enough
            else:
                logger.debug(
                    "filter_by_zone: %s intersects zone %s but does NOT match pattern %s",
                    det.label, zone_name, pattern,
                )

        if not matched:
            error_boxes.append(det.bbox)

    return kept, error_boxes


# ---------------------------------------------------------------------------
# Size filtering
# ---------------------------------------------------------------------------

def _parse_size_spec(spec: str, total_area: int) -> float:
    """Parse a size spec like ``"50%"`` or ``"300px"`` into absolute pixels."""
    m = re.match(r"(\d*\.?\d+)(px|%)?$", spec, re.IGNORECASE)
    if not m:
        logger.error("Invalid size spec: %s", spec)
        return 0.0
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "%":
        return value / 100.0 * total_area
    # Default (no unit or "px") -> absolute pixels
    return value


def filter_by_size(
    detections: list[Detection],
    max_size: str | None,
    image_shape: tuple[int, int],
) -> list[Detection]:
    """Filter detections whose area exceeds *max_size*.

    *max_size* may be ``"50%"`` (of image area) or ``"300px"`` (absolute
    pixel area).  If *max_size* is ``None`` or empty, all detections pass.
    """
    if not max_size:
        return detections

    h, w = image_shape
    max_area = _parse_size_spec(max_size, h * w)
    if max_area <= 0:
        return detections

    kept: list[Detection] = []
    for det in detections:
        if det.bbox.area > max_area:
            logger.debug(
                "filter_by_size: dropping %s (area %d > max %d)",
                det.label, det.bbox.area, int(max_area),
            )
        else:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Pattern filtering
# ---------------------------------------------------------------------------

def filter_by_pattern(
    detections: list[Detection],
    pattern: str,
) -> list[Detection]:
    """Keep only detections whose label matches *pattern* (regex)."""
    if not pattern or pattern == ".*":
        return detections

    compiled = re.compile(pattern)
    kept: list[Detection] = []
    for det in detections:
        if compiled.match(det.label):
            kept.append(det)
        else:
            logger.debug(
                "filter_by_pattern: dropping %s (does not match %s)",
                det.label, pattern,
            )
    return kept


# ---------------------------------------------------------------------------
# Past-detection filtering
# ---------------------------------------------------------------------------

def load_past_detections(past_file: str) -> tuple[list[list[int]], list[str]]:
    """Load ``(saved_boxes, saved_labels)`` from a pickle file.

    Returns ``([], [])`` on missing file, empty file, or read error.
    """
    import pickle  # lazy import

    try:
        with open(past_file, "rb") as fh:
            saved_boxes: list[list[int]] = pickle.load(fh)
            saved_labels: list[str] = pickle.load(fh)
        return saved_boxes, saved_labels
    except FileNotFoundError:
        logger.debug("No past-detection file found at %s", past_file)
    except EOFError:
        logger.debug("Empty past-detection file at %s, removing", past_file)
        try:
            os.remove(past_file)
        except OSError:
            pass
    except Exception:
        logger.exception("Error reading past detections from %s", past_file)
    return [], []


def save_past_detections(past_file: str, detections: list[Detection]) -> None:
    """Save current detections to a pickle file for future comparisons."""
    import pickle  # lazy import

    if not detections:
        return
    try:
        with open(past_file, "wb") as fh:
            pickle.dump([d.bbox.as_list() for d in detections], fh)
            pickle.dump([d.label for d in detections], fh)
        logger.debug("Saved %d detections to %s", len(detections), past_file)
    except Exception:
        logger.exception("Error saving past detections to %s", past_file)


def match_past_detections(
    detections: list[Detection],
    saved_boxes: list[list[int]],
    saved_labels: list[str],
    max_diff_area: str = "5%",
    label_area_overrides: dict[str, str] | None = None,
    ignore_labels: list[str] | None = None,
    aliases: list[list[str]] | None = None,
) -> list[Detection]:
    """Filter detections against past data.  Pure logic, no I/O.

    Parameters
    ----------
    saved_boxes, saved_labels:
        Previously saved detection data (from :func:`load_past_detections`).
    max_diff_area:
        Default area tolerance, e.g. ``"5%"`` or ``"300px"``.
    label_area_overrides:
        Per-label area tolerance, e.g. ``{"car": "10%"}``.
    ignore_labels:
        Labels to skip entirely (always kept, never matched).
    aliases:
        Groups of equivalent labels, e.g. ``[["car","bus","truck"]]``.
    """
    from shapely.geometry import Polygon  # optional dependency

    if not saved_boxes:
        return list(detections)

    label_area_overrides = label_area_overrides or {}
    ignore_labels = ignore_labels or []
    alias_map: dict[str, str] = {}
    for group in (aliases or []):
        canonical = group[0]
        for label in group:
            alias_map[label] = canonical

    kept: list[Detection] = []
    for det in detections:
        if det.label in ignore_labels:
            kept.append(det)
            continue

        det_poly = Polygon(det.bbox.as_polygon_coords())
        det_canonical = alias_map.get(det.label, det.label)
        found_match = False

        for saved_idx, saved_box in enumerate(saved_boxes):
            saved_canonical = alias_map.get(saved_labels[saved_idx], saved_labels[saved_idx])
            if saved_canonical != det_canonical:
                continue

            saved_bbox = BBox(x1=saved_box[0], y1=saved_box[1], x2=saved_box[2], y2=saved_box[3])
            saved_poly = Polygon(saved_bbox.as_polygon_coords())

            if not saved_poly.intersects(det_poly):
                continue

            if det_poly.contains(saved_poly):
                diff_area = det_poly.difference(saved_poly).area
                ref_area = det_poly.area
            else:
                diff_area = saved_poly.difference(det_poly).area
                ref_area = saved_poly.area

            effective_max = label_area_overrides.get(det.label, max_diff_area)
            max_pixels = _parse_size_spec(effective_max, int(ref_area)) if ref_area > 0 else 0

            if diff_area <= max_pixels:
                logger.debug(
                    "match_past_detections: %s at %s matches saved %s at %s (diff=%.0f <= max=%.0f), removing",
                    det.label, det.bbox, saved_labels[saved_idx], saved_box,
                    diff_area, max_pixels,
                )
                found_match = True
                break

        if not found_match:
            kept.append(det)

    return kept


def filter_past_detections(
    detections: list[Detection],
    past_file: str,
    max_diff_area: str = "5%",
    label_area_overrides: dict[str, str] | None = None,
    ignore_labels: list[str] | None = None,
    aliases: list[list[str]] | None = None,
) -> list[Detection]:
    """Compare detections with pickle-stored past detections and remove
    duplicates whose bounding-box area difference is within *max_diff_area*.

    Convenience wrapper around :func:`load_past_detections`,
    :func:`match_past_detections`, and :func:`save_past_detections`.

    After filtering, the current detections are saved back to *past_file*
    for future comparisons.
    """
    saved_boxes, saved_labels = load_past_detections(past_file)
    kept = match_past_detections(
        detections, saved_boxes, saved_labels,
        max_diff_area=max_diff_area,
        label_area_overrides=label_area_overrides,
        ignore_labels=ignore_labels,
        aliases=aliases,
    )
    save_past_detections(past_file, detections)
    return kept
