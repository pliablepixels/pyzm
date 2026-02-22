"""ZoneMinder event browser panel for the training UI.

Lets users connect to a ZM instance, browse monitors and events, preview
frames, and import selected frames as training images.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image as PILImage, ImageDraw
from st_clickable_images import clickable_images

from pyzm.train.dataset import YOLODataset
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")

_EVENT_THUMB_WIDTH = 640
_FRAME_THUMB_WIDTH = 640


# ------------------------------------------------------------------
# Placeholder image for failed loads
# ------------------------------------------------------------------

_PLACEHOLDER_BYTES: bytes | None = None


def _placeholder_image() -> bytes:
    global _PLACEHOLDER_BYTES
    if _PLACEHOLDER_BYTES is None:
        img = PILImage.new("RGB", (180, 120), "#2a2a2a")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "No preview", fill="#666")
        buf = BytesIO()
        img.save(buf, format="PNG")
        _PLACEHOLDER_BYTES = buf.getvalue()
    return _PLACEHOLDER_BYTES


def _to_data_uri(img_bytes: bytes) -> str:
    """Convert image bytes to a base64 data URI."""
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"


def _burn_caption(img_bytes: bytes, text: str) -> bytes:
    """Draw a small caption bar at the bottom of a thumbnail image."""
    try:
        from PIL import ImageFont

        img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
        w, h = img.size

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13,
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        bbox = font.getbbox(text)
        text_h = bbox[3] - bbox[1]
        bar_h = text_h + 8

        draw = ImageDraw.Draw(img)
        # Semi-transparent dark bar at the bottom
        bar_img = PILImage.new("RGBA", (w, bar_h), (0, 0, 0, 160))
        img_rgba = img.convert("RGBA")
        img_rgba.paste(bar_img, (0, h - bar_h), bar_img)
        draw = ImageDraw.Draw(img_rgba)

        text_w = bbox[2] - bbox[0]
        tx = (w - text_w) // 2
        ty = h - bar_h + 3
        draw.text((tx, ty), text, fill="#FFFFFF", font=font)

        buf = BytesIO()
        img_rgba.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return img_bytes


def _add_selection_overlay(img_bytes: bytes, color: tuple[int, ...] = (39, 174, 96), alpha: int = 80) -> bytes:
    """Apply a translucent green overlay on top of the thumbnail."""
    try:
        img = PILImage.open(BytesIO(img_bytes)).convert("RGBA")
        overlay = PILImage.new("RGBA", img.size, (*color, alpha))
        img = PILImage.alpha_composite(img, overlay)

        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return img_bytes


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def zm_event_browser_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args,
) -> None:
    """Top-level panel called from the 'Browse ZM Events' tab."""
    # Shared CSS
    st.markdown("""<style>
    .zm-section-header {
        font-size: 1.05em; font-weight: 700; margin: 8px 0 6px;
        padding: 6px 10px; border-left: 4px solid #2d6a4f;
        background: rgba(45,106,79,0.1); border-radius: 0 4px 4px 0;
    }
    </style>""", unsafe_allow_html=True)
    _zm_connection_form(ds.project_dir)

    if not st.session_state.get("zm_connected"):
        return

    zm = st.session_state["zm_client"]
    monitors = st.session_state.get("zm_monitors", [])

    monitor_id = _zm_monitor_picker(monitors)
    if monitor_id is None:
        return

    _zm_event_grid(zm, monitor_id)

    selected_event = st.session_state.get("zm_selected_event")
    if selected_event is not None:
        _zm_frame_grid(zm, selected_event, ds, store, args)


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------

def _load_zm_creds(project_dir: Path) -> dict:
    """Load saved ZM credentials from project.json."""
    meta_path = project_dir / "project.json"
    if not meta_path.exists():
        return {}
    try:
        import json
        meta = json.loads(meta_path.read_text())
        return meta.get("zm_connection", {})
    except Exception:
        return {}


def _save_zm_creds(project_dir: Path, url: str, user: str, password: str, verify_ssl: bool) -> None:
    """Save ZM credentials into project.json."""
    import json
    meta_path = project_dir / "project.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    meta["zm_connection"] = {
        "url": url,
        "user": user,
        "password": password,
        "verify_ssl": verify_ssl,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def _zm_connect(url: str, user: str, password: str, verify_ssl: bool) -> bool:
    """Try to connect to ZM. Returns True on success, sets session state."""
    from pyzm.client import ZMClient
    zm = ZMClient(
        url=url,
        user=user or None,
        password=password or None,
        verify_ssl=verify_ssl,
    )
    monitors = zm.monitors()
    st.session_state["zm_url"] = url
    st.session_state["zm_user"] = user
    st.session_state["zm_connected"] = True
    st.session_state["zm_client"] = zm
    st.session_state["zm_monitors"] = monitors
    return True


def _zm_connection_form(project_dir: Path) -> None:
    """Render ZM connection form with auto-connect from saved creds."""
    connected = st.session_state.get("zm_connected", False)

    # Auto-connect from saved creds on first load
    if not connected and "zm_auto_connect_tried" not in st.session_state:
        st.session_state["zm_auto_connect_tried"] = True
        saved = _load_zm_creds(project_dir)
        if saved.get("url"):
            try:
                _zm_connect(
                    saved["url"], saved.get("user", ""),
                    saved.get("password", ""), saved.get("verify_ssl", True),
                )
                st.rerun()
            except Exception as exc:
                logger.debug("Auto-connect with saved creds failed: %s", exc)
        connected = st.session_state.get("zm_connected", False)

    with st.expander(
        "ZM Connection" + (" -- Connected" if connected else ""),
        expanded=not connected,
    ):
        if connected:
            zm = st.session_state["zm_client"]
            version = zm.zm_version or "unknown"
            st.success(f"Connected to ZoneMinder {version}")
            if st.button("Disconnect"):
                _zm_disconnect()
                st.rerun()
            return

        saved = _load_zm_creds(project_dir)
        with st.form("zm_connect_form"):
            url = st.text_input(
                "ZM URL (portal or API)",
                value=saved.get("url", ""),
                placeholder="https://zm.example.com/zm",
                help="Portal URL or API URL -- both work.",
            )
            col_user, col_pass = st.columns(2)
            with col_user:
                user = st.text_input("Username", value=saved.get("user", "admin"), placeholder="admin")
            with col_pass:
                password = st.text_input("Password", value=saved.get("password", "admin"), type="password")
            verify_ssl = st.checkbox("Verify SSL", value=saved.get("verify_ssl", True))
            submitted = st.form_submit_button("Connect", type="primary")

        if submitted:
            if not url.strip():
                st.error("ZM URL is required.")
                return
            try:
                _zm_connect(url.strip(), user.strip(), password, verify_ssl)
                _save_zm_creds(project_dir, url.strip(), user.strip(), password, verify_ssl)
                st.rerun()
            except Exception as exc:
                st.error(f"Connection failed: {exc}")


def _zm_disconnect() -> None:
    """Clear all zm_* session state keys."""
    keys_to_clear = [k for k in st.session_state if k.startswith("zm_")]
    for k in keys_to_clear:
        del st.session_state[k]


# ------------------------------------------------------------------
# Frame image fetching
# ------------------------------------------------------------------

def _portal_url(zm) -> str:
    """Derive portal URL from ZMClient (strip /api suffix)."""
    api_url = zm.api.api_url
    if api_url.endswith("/api"):
        return api_url[:-4]
    return api_url


def _fetch_frame_image(zm, event_id: int, frame_id: int | str, width: int | None = None) -> bytes | None:
    """Fetch a single frame image from ZM's portal image endpoint."""
    base = _portal_url(zm)
    url = f"{base}/index.php?view=image&eid={event_id}&fid={frame_id}"
    if width:
        url += f"&width={width}"
    try:
        resp = zm.api.request(url)
        if hasattr(resp, "content") and resp.content:
            return resp.content
    except Exception as exc:
        logger.debug("Failed to fetch frame image eid=%s fid=%s: %s", event_id, frame_id, exc)
    return None


def _get_cached_thumbnails(
    zm, event_id: int, frame_meta: list,
    progress_cb: callable | None = None,
) -> dict[int | str, bytes]:
    """Fetch and cache thumbnails for frames in an event.

    Incrementally fetches thumbnails for any frames not already cached,
    so expanding the frame set (all alarm / custom IDs) works correctly.
    """
    cache_key = f"zm_thumbs_{event_id}"
    thumbnails: dict[int | str, bytes] = st.session_state.get(cache_key, {})

    missing = [fm for fm in frame_meta if fm.frame_id not in thumbnails]
    if missing:
        for i, fm in enumerate(missing):
            img_bytes = _fetch_frame_image(zm, event_id, fm.frame_id, width=_FRAME_THUMB_WIDTH)
            if img_bytes:
                thumbnails[fm.frame_id] = img_bytes
            if progress_cb:
                progress_cb(i + 1, len(missing))
        st.session_state[cache_key] = thumbnails

    return thumbnails


def _fetch_event_thumbnails(
    zm, events: list,
    progress_cb: callable | None = None,
) -> dict[int, bytes | None]:
    """Fetch snapshot thumbnails for a list of events. Cached in session state."""
    cache_key = "zm_event_thumbs"
    cached: dict[int, bytes | None] = st.session_state.get(cache_key, {})

    missing = [ev for ev in events if ev.id not in cached]
    if missing:
        for i, ev in enumerate(missing):
            fid = ev.max_score_frame_id or "snapshot"
            cached[ev.id] = _fetch_frame_image(zm, ev.id, fid, width=_EVENT_THUMB_WIDTH)
            if progress_cb:
                progress_cb(i + 1, len(missing))
        st.session_state[cache_key] = cached

    return cached


# ------------------------------------------------------------------
# Monitor picker
# ------------------------------------------------------------------

def _zm_monitor_picker(monitors) -> int | None:
    """Render a selectbox of monitors."""
    if not monitors:
        st.info("No monitors found on this ZM server.")
        return None

    options = {
        m.id: f"{m.id} - {m.name} ({m.function}, {m.width}x{m.height})"
        for m in monitors
    }
    return st.selectbox(
        "Monitor",
        options=list(options.keys()),
        format_func=lambda mid: options[mid],
        key="zm_selected_monitor",
    )


# ------------------------------------------------------------------
# Event grid (thumbnails)
# ------------------------------------------------------------------

_TIME_PRESETS = {
    "Last hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "Last 3 days": timedelta(days=3),
    "Last 7 days": timedelta(days=7),
    "Custom range": None,
}


def _zm_event_grid(zm, monitor_id: int) -> None:
    """Time filter + fetch + thumbnail grid of events."""
    today = datetime.now().date()

    col_preset, col_alarm, col_fetch = st.columns([2, 1, 1])
    with col_preset:
        time_preset = st.selectbox(
            "Time range",
            options=list(_TIME_PRESETS.keys()),
            index=4,  # "Last 7 days"
            key="zm_time_preset",
        )
    with col_alarm:
        min_alarm = st.number_input(
            "Min alarm frames", min_value=0, value=1, step=1, key="zm_min_alarm",
        )
    with col_fetch:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_clicked = st.button("Fetch Events", type="primary")

    # Show date pickers when "Custom range" is selected
    if time_preset == "Custom range":
        default_from = today - timedelta(days=7)
        col_from, col_to = st.columns(2)
        with col_from:
            date_from = st.date_input("From", value=default_from, key="zm_date_from")
        with col_to:
            date_to = st.date_input("To", value=today, key="zm_date_to")

    if fetch_clicked:
        if time_preset == "Custom range":
            since_str = datetime.combine(date_from, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
            until_str = datetime.combine(date_to, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")
        else:
            delta = _TIME_PRESETS[time_preset]
            since_str = (datetime.now() - delta).strftime("%Y-%m-%d %H:%M:%S")
            until_str = None
        try:
            events = zm.events(
                monitor_id=monitor_id,
                since=since_str,
                until=until_str,
                min_alarm_frames=min_alarm if min_alarm > 0 else None,
                limit=50,
            )
            st.session_state["zm_events"] = events
            st.session_state["zm_selected_event"] = None
            # Clear old event thumbnail cache
            st.session_state.pop("zm_event_thumbs", None)
        except Exception as exc:
            st.error(f"Failed to fetch events: {exc}")
            return

    events = st.session_state.get("zm_events")
    if not events:
        return

    # Fetch thumbnails for events
    _evt_prog = st.progress(0, text="Loading event thumbnails...")
    def _evt_cb(done: int, total: int) -> None:
        _evt_prog.progress(done / total, text=f"Loading event thumbnails... {done}/{total}")
    event_thumbs = _fetch_event_thumbnails(zm, events, progress_cb=_evt_cb)
    _evt_prog.empty()

    selected_eid = st.session_state.get("zm_selected_event")
    placeholder = _placeholder_image()

    paths = []
    titles = []
    for ev in events:
        thumb = event_thumbs.get(ev.id) or placeholder
        start = ev.start_time.strftime("%b %d %H:%M") if ev.start_time else "?"
        caption = f"Event {ev.id}  |  {start}  |  score {ev.max_score}"
        thumb = _burn_caption(thumb, caption)
        is_selected = ev.id == selected_eid
        if is_selected:
            thumb = _add_selection_overlay(thumb)
        paths.append(_to_data_uri(thumb))
        titles.append(caption)

    clicked_idx = clickable_images(
        paths, titles=titles,
        div_style={"display": "flex", "flex-wrap": "wrap", "justify-content": "flex-start"},
        img_style={
            "width": "23%", "margin": "1%", "border-radius": "6px", "cursor": "pointer",
            "border": "2px solid transparent",
        },
        key=f"zm_event_grid_{selected_eid}",
    )
    if clicked_idx > -1 and clicked_idx < len(events):
        new_eid = events[clicked_idx].id
        if new_eid != selected_eid:
            st.session_state["zm_selected_event"] = new_eid
            st.rerun()


# ------------------------------------------------------------------
# Frame helpers
# ------------------------------------------------------------------

def _initial_key_frames(frame_meta: list, event_id: int) -> list:
    """Return the 2 key frames: first alarm frame + snapshot (max-score) frame."""
    if not frame_meta:
        return []

    snapshot_fid = None
    events = st.session_state.get("zm_events", [])
    for ev in events:
        if ev.id == event_id and ev.max_score_frame_id is not None:
            snapshot_fid = ev.max_score_frame_id
            break

    first_alarm = next((f for f in frame_meta if f.type == "Alarm"), None)

    result = []
    seen = set()
    if first_alarm and first_alarm.frame_id not in seen:
        result.append(first_alarm)
        seen.add(first_alarm.frame_id)
    for f in frame_meta:
        if f.frame_id == snapshot_fid and f.frame_id not in seen:
            result.append(f)
            seen.add(f.frame_id)
            break

    return result


def _all_alarm_frames(frame_meta: list) -> list:
    """Return all Alarm-type frames."""
    return [f for f in frame_meta if f.type == "Alarm"]


def _custom_frames(frame_meta: list, frame_ids: list[int]) -> list:
    """Return frames matching the given IDs."""
    id_set = set(frame_ids)
    return [f for f in frame_meta if f.frame_id in id_set]


# ------------------------------------------------------------------
# Frame grid
# ------------------------------------------------------------------

def _zm_frame_grid(zm, event_id: int, ds: YOLODataset, store: VerificationStore, args) -> None:
    """Frame grid with thumbnails, checkboxes, and import."""
    st.divider()

    all_key = f"zm_all_frames_{event_id}"
    if all_key not in st.session_state:
        try:
            ev = zm.event(event_id)
            st.session_state[all_key] = ev.get_frames()
        except Exception as exc:
            st.error(f"Failed to fetch frames: {exc}")
            return
    all_frames = st.session_state[all_key]

    meta_key = f"zm_frames_{event_id}"
    if meta_key not in st.session_state:
        st.session_state[meta_key] = _initial_key_frames(all_frames, event_id)
    frame_meta = st.session_state[meta_key]

    total_alarm = sum(1 for f in all_frames if f.type == "Alarm")

    if not frame_meta:
        st.info("No alarm or snapshot frames in this event.")
        return

    st.markdown(
        f"<div class='zm-section-header'>"
        f"Select frames to audit for Event {event_id} "
        f"<span style='font-weight:400; font-size:0.85em;'>"
        f"({len(all_frames)} total frames, {total_alarm} alarm)</span></div>",
        unsafe_allow_html=True,
    )

    # Load more options
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if total_alarm > len(frame_meta):
            if st.button(f"All {total_alarm} alarm", key=f"zm_load_alarm_{event_id}"):
                alarm = _all_alarm_frames(all_frames)
                merged = {f.frame_id: f for f in frame_meta}
                for f in alarm:
                    merged[f.frame_id] = f
                st.session_state[meta_key] = sorted(merged.values(), key=lambda f: f.frame_id)
                st.rerun()
    with c2:
        custom_ids = st.text_input(
            "Select custom frames by ID",
            placeholder="5,10,42",
            key=f"zm_custom_ids_{event_id}",
        )
    with c3:
        if custom_ids and custom_ids.strip():
            if st.button("Load", key=f"zm_load_custom_{event_id}"):
                try:
                    ids = [int(x.strip()) for x in custom_ids.split(",") if x.strip()]
                except ValueError:
                    st.error("Comma-separated numbers only")
                else:
                    custom = _custom_frames(all_frames, ids)
                    merged = {f.frame_id: f for f in frame_meta}
                    for f in custom:
                        merged[f.frame_id] = f
                    st.session_state[meta_key] = sorted(merged.values(), key=lambda f: f.frame_id)
                    st.rerun()

    # Thumbnails
    _frm_prog = st.progress(0, text="Loading frame thumbnails...")
    def _frm_cb(done: int, total: int) -> None:
        _frm_prog.progress(done / total, text=f"Loading frame thumbnails... {done}/{total}")
    thumbnails = _get_cached_thumbnails(zm, event_id, frame_meta, progress_cb=_frm_cb)
    _frm_prog.empty()

    # Frame thumbnail grid — click to import directly
    placeholder = _placeholder_image()
    paths = []
    titles = []
    for fm in frame_meta:
        thumb_bytes = thumbnails.get(fm.frame_id) or placeholder
        paths.append(_to_data_uri(thumb_bytes))
        frame_type = "Alarm" if fm.type == "Alarm" else "Snap"
        titles.append(f"Frame {fm.frame_id} | {frame_type} | score {fm.score}")

    # Counter in key prevents stale clicks from re-importing on rerun
    import_counter = st.session_state.get("_frame_import_counter", 0)
    clicked_idx = clickable_images(
        paths, titles=titles,
        div_style={"display": "flex", "flex-wrap": "wrap", "justify-content": "flex-start"},
        img_style={
            "width": "23%", "margin": "1%", "border-radius": "6px", "cursor": "pointer",
            "border": "2px solid transparent",
        },
        key=f"zm_frame_grid_{event_id}_{import_counter}",
    )
    if clicked_idx > -1 and clicked_idx < len(frame_meta):
        fid = frame_meta[clicked_idx].frame_id
        st.session_state["_frame_import_counter"] = import_counter + 1
        _import_frames(ds, store, args, zm, event_id, [fid])
        st.rerun()


# ------------------------------------------------------------------
# Import pipeline
# ------------------------------------------------------------------

def _import_frames(
    ds: YOLODataset,
    store: VerificationStore,
    args,
    zm,
    event_id: int,
    frame_ids: list[int | str],
) -> None:
    """Fetch frame images from ZM, save to disk, add to dataset."""
    from pyzm.models.config import StreamConfig

    progress = st.progress(0, text="Importing frames...")
    imported = 0

    sc = StreamConfig(
        frame_set=[str(fid) for fid in frame_ids],
        resize=None,
    )
    try:
        ev = zm.event(event_id)
        frames, _dims = ev.extract_frames(stream_config=sc)
    except Exception as exc:
        st.error(f"Failed to fetch frame images: {exc}")
        return

    for i, (fid, img_array) in enumerate(frames):
        tmp_dir = Path(tempfile.mkdtemp())
        fname = f"event{event_id}_frame{fid}.jpg"
        tmp_path = tmp_dir / fname

        try:
            import cv2
            cv2.imwrite(str(tmp_path), img_array)
        except Exception:
            pil_img = PILImage.fromarray(
                img_array if img_array.shape[2] == 3 else img_array[..., :3]
            )
            pil_img.save(str(tmp_path), "JPEG")

        dest = ds.add_image(tmp_path, [])
        imported += 1

        store.set(ImageVerification(
            image_name=dest.name,
            detections=[],
            fully_reviewed=False,
        ))
        progress.progress((i + 1) / len(frames), text=f"Imported {i + 1}/{len(frames)}")

    store.save()
    progress.progress(1.0, text=f"Done: {imported} frames imported")
    st.toast(f"Imported {imported} frames")

    # Auto-navigate to review phase
    if imported > 0:
        st.session_state["active_phase"] = "review"
