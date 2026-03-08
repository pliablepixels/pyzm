"""Shared-memory reader for ZoneMinder monitor status.

Reads the ``SharedData`` and ``TriggerData`` structs that ZoneMinder writes
to ``/dev/shm/zm.mmap.<monitor_id>``.  Supports both ZM 1.36.x and 1.38+
struct formats, auto-detected from the ``size`` field at offset 0.

This is a clean rewrite of the legacy ``ZMMemory`` class with:

* No global state -- everything is instance-based.
* Python 3.10+ syntax.
* Proper resource management via context-manager protocol.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from collections import namedtuple

logger = logging.getLogger("pyzm.zm")


# ---------------------------------------------------------------------------
# Struct layouts
# ---------------------------------------------------------------------------
# Each entry is ``(field_name, struct_format_char)``.
# Format chars ending with ``'s'`` (e.g. ``'256s'``) are byte-string fields.

# ZM <= 1.36.33
_LAYOUT_760 = [
    ("size", "I"), ("last_write_index", "i"), ("last_read_index", "i"),
    ("state", "I"),
    ("capture_fps", "d"), ("analysis_fps", "d"),
    ("last_event", "Q"), ("action", "I"),
    ("brightness", "i"), ("hue", "i"), ("color", "i"), ("contrast", "i"),
    ("alarm_x", "i"), ("alarm_y", "i"),
    ("valid", "?"), ("active", "?"), ("signal", "?"), ("format", "?"),
    ("imagesize", "I"), ("last_frame_score", "I"),
    ("audio_frequency", "I"), ("audio_channels", "I"),
    ("startup_time", "q"), ("heartbeat_time", "q"),
    ("last_write_time", "q"), ("last_read_time", "q"),
    ("control_state", "256s"), ("alarm_cause", "256s"),
    ("video_fifo", "64s"), ("audio_fifo", "64s"),
]

# ZM 1.36.34+ (added image_count after last_read_index — ZM commit 194b55a)
_LAYOUT_764 = [
    ("size", "I"), ("last_write_index", "i"), ("last_read_index", "i"),
    ("image_count", "i"),
    ("state", "I"),
    ("capture_fps", "d"), ("analysis_fps", "d"),
    ("last_event", "Q"), ("action", "I"),
    ("brightness", "i"), ("hue", "i"), ("color", "i"), ("contrast", "i"),
    ("alarm_x", "i"), ("alarm_y", "i"),
    ("valid", "?"), ("active", "?"), ("signal", "?"), ("format", "?"),
    ("imagesize", "I"), ("last_frame_score", "I"),
    ("audio_frequency", "I"), ("audio_channels", "I"),
    ("startup_time", "q"), ("heartbeat_time", "q"),
    ("last_write_time", "q"), ("last_read_time", "q"),
    ("control_state", "256s"), ("alarm_cause", "256s"),
    ("video_fifo", "64s"), ("audio_fifo", "64s"),
]

# ZM 1.38+
_LAYOUT_872 = [
    ("size", "I"), ("last_write_index", "i"), ("last_read_index", "i"),
    ("image_count", "i"), ("state", "I"),
    ("capture_fps", "d"), ("analysis_fps", "d"),
    ("latitude", "d"), ("longitude", "d"),
    ("last_event", "Q"), ("action", "I"),
    ("brightness", "i"), ("hue", "i"), ("color", "i"), ("contrast", "i"),
    ("alarm_x", "i"), ("alarm_y", "i"),
    ("valid", "?"), ("capturing", "?"), ("analysing", "?"),
    ("recording", "?"), ("signal", "?"), ("format", "?"),
    ("reserved1", "?"), ("reserved2", "?"),
    ("imagesize", "I"), ("last_frame_score", "I"),
    ("audio_frequency", "I"), ("audio_channels", "I"),
    ("startup_time", "q"), ("heartbeat_time", "q"),
    ("last_write_time", "q"), ("last_read_time", "q"),
    ("last_viewed_time", "q"), ("last_analysis_viewed_time", "q"),
    ("control_state", "256s"), ("alarm_cause", "256s"),
    ("video_fifo", "64s"), ("audio_fifo", "64s"), ("janus_pin", "64s"),
]


def _build_struct_info(
    layout: list[tuple[str, str]],
) -> tuple[str, list[str], list[str]]:
    """Derive ``(struct_fmt, field_names, string_fields)`` from a layout."""
    fmt = "@" + "".join(char for _, char in layout)
    fields = [name for name, _ in layout]
    string_fields = [name for name, char in layout if char.endswith("s")]
    return fmt, fields, string_fields


# Registry keyed by struct size.  Adding a new ZM version is a single
# ``_LAYOUT_xxx`` appended to this list.
_REGISTRY: dict[int, tuple[str, list[str], list[str]]] = {}
for _layout in [_LAYOUT_760, _LAYOUT_764, _LAYOUT_872]:
    _fmt, _fields, _str_fields = _build_struct_info(_layout)
    _REGISTRY[struct.calcsize(_fmt)] = (_fmt, _fields, _str_fields)


# TriggerData -- unchanged across ZM versions.
_TRIGGER_FMT = "IIII32s256s256s"
_TRIGGER_FIELDS = [
    "size", "trigger_state", "trigger_score", "padding",
    "trigger_cause", "trigger_text", "trigger_showtext",
]
_TRIGGER_SIZE = struct.calcsize(_TRIGGER_FMT)
_TRIGGER_STRING_FIELDS = ["trigger_cause", "trigger_text", "trigger_showtext"]
_TriggerData = namedtuple("TriggerData", _TRIGGER_FIELDS)


# ---------------------------------------------------------------------------
# Alarm state constants
# ---------------------------------------------------------------------------

ALARM_STATES: dict[str, int] = {
    "STATE_IDLE": 0,
    "STATE_PREALARM": 1,
    "STATE_ALARM": 2,
    "STATE_ALERT": 3,
    "STATE_TAPE": 4,
    "ACTION_GET": 5,
    "ACTION_SET": 6,
    "ACTION_RELOAD": 7,
    "ACTION_SUSPEND": 8,
    "ACTION_RESUME": 9,
    "TRIGGER_CANCEL": 10,
    "TRIGGER_ON": 11,
    "TRIGGER_OFF": 12,
}

# Reverse lookup: int -> name
_STATE_NAMES: dict[int, str] = {v: k for k, v in ALARM_STATES.items()}


# ---------------------------------------------------------------------------
# SharedMemory class
# ---------------------------------------------------------------------------

class SharedMemory:
    """Read ZoneMinder shared memory for a single monitor.

    Parameters
    ----------
    monitor_id:
        The numeric ZM monitor ID.
    shm_path:
        Directory where ZM mmap files live (default ``/dev/shm``).

    Raises
    ------
    ValueError
        If *monitor_id* is ``None`` or the mmap file is empty / corrupt.
    """

    def __init__(self, monitor_id: int, shm_path: str = "/dev/shm") -> None:
        if monitor_id is None:
            raise ValueError("No monitor ID specified")

        self._monitor_id = monitor_id
        self._mmap_path = os.path.join(shm_path, f"zm.mmap.{monitor_id}")
        self._fhandle: object | None = None
        self._mhandle: mmap.mmap | None = None
        self._layout: tuple[str, list[str], list[str]] | None = None

        self._open()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> SharedMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Close and re-open the mmap.  Call after external state changes."""
        self.close()
        self._layout = None
        self._open()

    def is_valid(self) -> bool:
        """``True`` if the shared memory can be read and ``size != 0``."""
        try:
            data = self._read()
            return data["shared_data"]["size"] != 0
        except Exception as exc:
            logger.error("Validity check failed: %s", exc)
            return False

    def is_alarmed(self) -> bool:
        """``True`` if the monitor is currently in ALARM state."""
        data = self._read()
        return int(data["shared_data"]["state"]) == ALARM_STATES["STATE_ALARM"]

    def alarm_state(self) -> dict[str, int | str]:
        """Return the current alarm state as ``{"id": int, "state": str}``."""
        data = self._read()
        state_id = int(data["shared_data"]["state"])
        return {
            "id": state_id,
            "state": _STATE_NAMES.get(state_id, f"UNKNOWN_{state_id}"),
        }

    def last_event(self) -> int:
        """Return the last event ID recorded by this monitor."""
        data = self._read()
        return data["shared_data"]["last_event"]

    def cause(self) -> dict[str, str | None]:
        """Return alarm and trigger causes."""
        data = self._read()
        return {
            "alarm_cause": data["shared_data"].get("alarm_cause"),
            "trigger_cause": data["trigger_data"].get("trigger_cause"),
        }

    def trigger(self) -> dict[str, str | dict | None]:
        """Return trigger information."""
        data = self._read()
        td = data["trigger_data"]
        state_id = td.get("trigger_state", 0)
        return {
            "trigger_text": td.get("trigger_text"),
            "trigger_showtext": td.get("trigger_showtext"),
            "trigger_cause": td.get("trigger_cause"),
            "trigger_state": {
                "id": state_id,
                "state": _STATE_NAMES.get(int(state_id), f"UNKNOWN_{state_id}"),
            },
        }

    def get(self) -> dict[str, dict]:
        """Return raw shared and trigger data dicts."""
        return self._read()

    def get_shared_data(self) -> dict:
        """Return just the shared data dict."""
        return self._read()["shared_data"]

    def get_trigger_data(self) -> dict:
        """Return just the trigger data dict."""
        return self._read()["trigger_data"]

    def close(self) -> None:
        """Close file and mmap handles (safe to call multiple times)."""
        if self._mhandle is not None:
            try:
                self._mhandle.close()
            except Exception:
                pass
            self._mhandle = None

        if self._fhandle is not None:
            try:
                self._fhandle.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._fhandle = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the mmap file and read the initial data."""
        file_size = os.path.getsize(self._mmap_path)
        if not file_size:
            raise ValueError(
                f"Empty mmap file ({file_size} bytes): {self._mmap_path}"
            )

        self._fhandle = open(self._mmap_path, "r+b")  # noqa: SIM115
        self._mhandle = mmap.mmap(
            self._fhandle.fileno(), 0, access=mmap.ACCESS_READ,  # type: ignore[union-attr]
        )
        # Eagerly detect the layout so errors surface immediately
        self._detect_layout()

    def _detect_layout(self) -> None:
        """Auto-detect the SharedData struct format from the ``size`` field."""
        assert self._mhandle is not None
        self._mhandle.seek(0)
        size_val = struct.unpack("@I", self._mhandle.read(4))[0]

        if size_val not in _REGISTRY:
            raise ValueError(
                f"Unknown SharedData size {size_val} in {self._mmap_path}. "
                f"Expected one of: {sorted(_REGISTRY.keys())}"
            )
        self._layout = _REGISTRY[size_val]

    def _read(self) -> dict[str, dict]:
        """Read current SharedData and TriggerData from the mmap."""
        if self._layout is None:
            self._detect_layout()

        assert self._mhandle is not None
        assert self._layout is not None
        struct_fmt, fields, string_fields = self._layout

        self._mhandle.seek(0)
        struct_size = struct.calcsize(struct_fmt)
        if not hasattr(self, '_SharedData') or self._SharedData._fields != tuple(fields):
            self._SharedData = namedtuple("SharedData", fields)
        SharedData = self._SharedData
        raw_sd = SharedData._make(
            struct.unpack(struct_fmt, self._mhandle.read(struct_size)),
        )

        TriggerData = _TriggerData
        raw_td = TriggerData._make(
            struct.unpack(_TRIGGER_FMT, self._mhandle.read(_TRIGGER_SIZE)),
        )

        sd = raw_sd._asdict()
        td = raw_td._asdict()

        # Decode null-terminated byte strings
        for key in string_fields:
            sd[key] = sd[key].split(b"\0", 1)[0].decode(errors="replace")
        for key in _TRIGGER_STRING_FIELDS:
            td[key] = td[key].split(b"\0", 1)[0].decode(errors="replace")

        # Backward compat: ZM 1.38 renamed "active" to "capturing"
        if "capturing" in sd:
            sd["active"] = sd["capturing"]

        return {"shared_data": sd, "trigger_data": td}
