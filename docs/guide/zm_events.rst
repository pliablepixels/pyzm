Events & Frames
================

Querying events
----------------

.. code-block:: python

   # Events from the last hour
   events = zm.events(since="1 hour ago", limit=5)
   for ev in events:
       print(f"Event {ev.id}: {ev.cause} ({ev.length:.1f}s, {ev.alarm_frames} alarm frames)")

   # Single event
   ev = zm.event(12345)

Filters: ``event_id``, ``monitor_id``, ``since``, ``until``,
``min_alarm_frames``, ``object_only``, ``limit``.

Per-frame metadata
-------------------

.. code-block:: python

   ev = zm.event(12345)
   frames = ev.get_frames()
   for f in frames:
       print(f"Frame {f.frame_id}: type={f.type} score={f.score} delta={f.delta:.2f}s")

   # Find the highest-scoring frame
   best = max(frames, key=lambda f: f.score)
   print(f"Best frame: {best.frame_id} (score={best.score})")

``get_frames()`` returns ``list[Frame]`` with per-frame ``Score``,
``Type`` (Normal/Alarm/Bulk), and ``Delta`` (seconds since event start).

Event management
-----------------

.. code-block:: python

   # Delete a single event
   ev = zm.event(12345)
   ev.delete()

   # Bulk delete events matching filters
   count = zm.delete_events(monitor_id=1, before="7 days ago", limit=500)
   print(f"Deleted {count} events")

   # OOP: query events scoped to a monitor
   m = zm.monitor(1)
   recent = m.events(until="6 hours ago", limit=10)

   # OOP: bulk-delete old events for a monitor
   count = m.delete_events(before="1 week ago", limit=500)
   print(f"Deleted {count} events from {m.name}")

Accessing the full API response
--------------------------------

pyzm models only extract a subset of the fields returned by the ZM API.
Every API-sourced object carries a ``raw()`` method that returns the full,
unmodified API response dict — useful for accessing fields like ``Path``,
``Protocol``, ``StorageId``, etc.:

.. code-block:: python

   m = zm.monitor(1)
   m.raw()["Monitor"]["Path"]        # e.g. "rtsp://cam/stream"
   m.raw()["Monitor"]["Protocol"]    # e.g. "rtsp"
   m.status.raw()                    # full Monitor_Status sub-dict

   ev = zm.event(12345)
   ev.raw()["Event"]["DiskSpace"]    # field not on the Event dataclass

   frames = ev.get_frames()
   frames[0].raw()                   # full Frame API dict

   zones = m.get_zones()
   zones[0].raw()                    # full Zone API dict including AlarmRGB, etc.

``raw()`` is available on ``Monitor``, ``MonitorStatus``, ``Event``,
``Frame``, ``Zone``, and ``PTZCapabilities``.
