"""ZoneMinder data models.

Typed representations of ZM resources (monitors, events, frames, zones).
Models carry a ``_client`` back-reference so that resource operations
(arm, disarm, zones, frames, etc.) are methods on the object itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyzm.client import ZMClient
    from pyzm.models.config import StreamConfig


@dataclass
class PTZCapabilities:
    """PTZ control profile capabilities."""
    # Movement
    can_move: bool = False
    can_move_con: bool = False
    can_move_rel: bool = False
    can_move_abs: bool = False
    can_move_diag: bool = False
    # Zoom
    can_zoom: bool = False
    can_zoom_con: bool = False
    can_zoom_rel: bool = False
    can_zoom_abs: bool = False
    # Pan/Tilt
    can_pan: bool = False
    can_tilt: bool = False
    # Presets
    has_presets: bool = False
    num_presets: int = 0
    has_home_preset: bool = False
    can_set_presets: bool = False
    # Other
    can_reset: bool = False

    @classmethod
    def from_api_dict(cls, data: dict) -> PTZCapabilities:
        """Build from a ZM API ``Control`` JSON dict."""
        def _bool(key: str) -> bool:
            return str(data.get(key, "0")) == "1"

        return cls(
            can_move=_bool("CanMove"),
            can_move_con=_bool("CanMoveCon"),
            can_move_rel=_bool("CanMoveRel"),
            can_move_abs=_bool("CanMoveAbs"),
            can_move_diag=_bool("CanMoveDiag"),
            can_zoom=_bool("CanZoom"),
            can_zoom_con=_bool("CanZoomCon"),
            can_zoom_rel=_bool("CanZoomRel"),
            can_zoom_abs=_bool("CanZoomAbs"),
            can_pan=_bool("CanPan"),
            can_tilt=_bool("CanTilt"),
            has_presets=_bool("HasPresets"),
            num_presets=int(data.get("NumPresets", 0) or 0),
            has_home_preset=_bool("HasHomePreset"),
            can_set_presets=_bool("CanSetPresets"),
            can_reset=_bool("CanReset"),
        )


@dataclass
class Zone:
    """A detection zone polygon belonging to a monitor."""
    name: str
    points: list[tuple[int, int]]
    pattern: str | None = None
    ignore_pattern: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.points,
            "pattern": self.pattern,
            "ignore_pattern": self.ignore_pattern,
        }


@dataclass
class Frame:
    """Metadata for a single event frame."""
    frame_id: int | str
    event_id: int
    type: str = ""        # "Alarm", "Bulk", "Normal", etc.
    score: int = 0
    delta: float = 0.0


@dataclass
class Event:
    """A ZoneMinder event.

    When retrieved via :class:`ZMClient`, the event carries a back-reference
    so that resource operations are available as methods::

        ev = zm.event(12345)
        ev.update_notes("person detected")
        ev.tag(["person", "car"])
        ev.delete()
    """
    id: int
    name: str = ""
    monitor_id: int = 0
    cause: str = ""
    notes: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    length: float = 0.0
    frames: int = 0
    alarm_frames: int = 0
    max_score: int = 0
    max_score_frame_id: int | None = None
    storage_path: str = ""

    _client: ZMClient | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Resource methods (delegate to ZMClient internals)
    # ------------------------------------------------------------------

    def get_frames(self) -> list[Frame]:
        """Fetch per-frame metadata (Score, Type, Delta) for this event."""
        self._require_client()
        return self._client._event_frames(self.id)

    def update_notes(self, notes: str) -> None:
        """Update the Notes field of this event."""
        self._require_client()
        self._client._update_event_notes(self.id, notes)

    def tag(self, labels: list[str]) -> None:
        """Tag this event with detected object labels."""
        self._require_client()
        self._client._tag_event(self.id, labels)

    def delete(self) -> None:
        """Delete this event."""
        self._require_client()
        self._client._delete_event(self.id)

    def path(self) -> str | None:
        """Construct the filesystem path for this event."""
        self._require_client()
        return self._client._event_path(self.id)

    def extract_frames(
        self, stream_config: StreamConfig | None = None,
    ) -> tuple[list[tuple[int | str, Any]], dict[str, tuple[int, int] | None]]:
        """Extract frames from this event as numpy arrays."""
        self._require_client()
        return self._client._get_event_frames(self.id, stream_config)

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "Event not bound to a ZMClient. "
                "Use zm.event() or zm.events() to get bound Event objects."
            )

    @classmethod
    def from_api_dict(cls, data: dict, client: ZMClient | None = None) -> Event:
        """Build from a ZM API ``Event`` JSON dict."""
        ev = data.get("Event", data)
        return cls(
            id=int(ev.get("Id", 0)),
            name=ev.get("Name", ""),
            monitor_id=int(ev.get("MonitorId", 0)),
            cause=ev.get("Cause", ""),
            notes=ev.get("Notes", ""),
            start_time=_parse_dt(ev.get("StartTime")),
            end_time=_parse_dt(ev.get("EndTime")),
            length=float(ev.get("Length", 0)),
            frames=int(ev.get("Frames", 0)),
            alarm_frames=int(ev.get("AlarmFrames", 0)),
            max_score=int(ev.get("MaxScore", 0)),
            max_score_frame_id=int(ev["MaxScoreFrameId"]) if ev.get("MaxScoreFrameId") else None,
            storage_path=ev.get("StoragePath", ""),
            _client=client,
        )


@dataclass
class MonitorStatus:
    """Runtime status of a monitor."""
    state: str = ""          # "Idle", "Alarm", etc.
    fps: float = 0.0
    analysis_fps: float = 0.0
    bandwidth: int = 0       # CaptureBandwidth in bytes
    capturing: str = "None"  # "None", "Capturing", etc.


@dataclass
class Monitor:
    """A ZoneMinder monitor.

    When retrieved via :class:`ZMClient`, the monitor carries a back-reference
    so that resource operations are available as methods::

        m = zm.monitor(1)
        m.zones()
        m.arm()
        m.update(Function="Modect")
    """
    id: int
    name: str = ""
    function: str = ""      # "Monitor", "Modect", "Record", etc.
    enabled: bool = True
    width: int = 0
    height: int = 0
    type: str = ""           # "Local", "Remote", "File", "Ffmpeg", etc.
    controllable: bool = False
    control_id: int | None = None
    zones: list[Zone] = field(default_factory=list)
    status: MonitorStatus = field(default_factory=MonitorStatus)

    _client: ZMClient | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Resource methods (delegate to ZMClient internals)
    # ------------------------------------------------------------------

    def get_zones(self) -> list[Zone]:
        """Fetch detection zones for this monitor from the ZM API."""
        self._require_client()
        return self._client._monitor_zones(self.id)

    def arm(self) -> dict:
        """Trigger alarm ON for this monitor."""
        self._require_client()
        return self._client._arm(self.id)

    def disarm(self) -> dict:
        """Cancel alarm for this monitor."""
        self._require_client()
        return self._client._disarm(self.id)

    def alarm_status(self) -> dict:
        """Get alarm status for this monitor."""
        self._require_client()
        return self._client._alarm_status(self.id)

    def update(self, **kwargs) -> None:
        """Update monitor fields (Function, Enabled, Name, etc.)."""
        self._require_client()
        self._client._update_monitor(self.id, **kwargs)

    def daemon_status(self) -> dict:
        """Get daemon status for this monitor's capture daemon (zmc)."""
        self._require_client()
        return self._client._daemon_status(self.id)

    def streaming_url(self, protocol: str = "mjpeg", **kwargs) -> str:
        """Return a streaming URL for this monitor.

        Parameters
        ----------
        protocol:
            Streaming protocol.  Currently only ``"mjpeg"`` is supported.
        **kwargs:
            Extra query params forwarded to the URL (e.g. ``maxfps=5``,
            ``scale=50``).
        """
        self._require_client()
        if protocol != "mjpeg":
            raise ValueError(f"Unknown streaming protocol: {protocol!r}")
        return self._client._streaming_url_mjpeg(self.id, **kwargs)

    def snapshot_url(self, **kwargs) -> str:
        """Return a single-frame snapshot URL for this monitor.

        Parameters
        ----------
        **kwargs:
            Extra query params forwarded to the URL (e.g. ``scale=50``).
        """
        self._require_client()
        return self._client._snapshot_url(self.id, **kwargs)

    def events(self, **kwargs) -> list[Event]:
        """Query events scoped to this monitor.

        Accepts the same filter params as :meth:`ZMClient.events`:
        ``since``, ``until``, ``min_alarm_frames``, ``object_only``, ``limit``.
        """
        self._require_client()
        return self._client.events(monitor_id=self.id, **kwargs)

    def delete_events(self, **kwargs) -> int:
        """Bulk-delete events scoped to this monitor.

        Accepts the same filter params as :meth:`ZMClient.delete_events`:
        ``before``, ``min_alarm_frames``, ``limit``.
        """
        self._require_client()
        return self._client.delete_events(monitor_id=self.id, **kwargs)

    def ptz_capabilities(self) -> PTZCapabilities:
        """Fetch PTZ capabilities for this monitor's control profile."""
        self._require_client()
        if not self.control_id:
            raise ValueError(
                f"Monitor {self.id} ({self.name!r}) has no control profile"
            )
        return self._client._ptz_capabilities(self.control_id)

    def ptz(
        self, command: str, *, mode: str = "con", preset: int = 1,
        stop_after: float | None = None,
    ) -> None:
        """Send a PTZ command to this monitor.

        Parameters
        ----------
        command:
            Direction or action: ``"up"``, ``"down"``, ``"left"``,
            ``"right"``, ``"up_left"``, ``"up_right"``, ``"down_left"``,
            ``"down_right"``, ``"zoom_in"``, ``"zoom_out"``, ``"stop"``,
            ``"home"``, ``"preset"``.
        mode:
            Movement mode — ``"con"`` (continuous, default), ``"rel"``
            (relative), or ``"abs"`` (absolute).
        preset:
            Preset number for ``command="preset"`` (default 1).
        stop_after:
            If set, automatically send a ``"stop"`` command after this
            many seconds.  Blocks for the duration.
        """
        self._require_client()
        if not self.controllable:
            raise ValueError(
                f"Monitor {self.id} ({self.name!r}) is not controllable"
            )
        self._client._ptz_command(
            self.id, command, mode=mode, preset=preset, stop_after=stop_after,
        )

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "Monitor not bound to a ZMClient. "
                "Use zm.monitor() or zm.monitors() to get bound Monitor objects."
            )

    @classmethod
    def from_api_dict(cls, data: dict, client: ZMClient | None = None) -> Monitor:
        """Build from a ZM API ``Monitor`` JSON dict."""
        mon = data.get("Monitor", data)
        status_data = data.get("Monitor_Status", {})
        raw_control_id = mon.get("ControlId")
        return cls(
            id=int(mon.get("Id", 0)),
            name=mon.get("Name", ""),
            function=mon.get("Function", ""),
            enabled=mon.get("Enabled") == "1",
            width=int(mon.get("Width", 0)),
            height=int(mon.get("Height", 0)),
            type=mon.get("Type", ""),
            controllable=mon.get("Controllable") == "1",
            control_id=int(raw_control_id) if raw_control_id else None,
            status=MonitorStatus(
                state=status_data.get("Status", ""),
                fps=float(status_data.get("CaptureFPS", 0) or 0),
                analysis_fps=float(status_data.get("AnalysisFPS", 0) or 0),
                bandwidth=int(status_data.get("CaptureBandwidth", 0) or 0),
                capturing=status_data.get("Capturing", "None"),
            ),
            _client=client,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(val, fmt)
        except (ValueError, TypeError):
            continue
    return None
