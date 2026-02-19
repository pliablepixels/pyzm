Monitors & Zones
=================

Listing monitors
-----------------

.. code-block:: python

   for m in zm.monitors():
       print(f"Monitor {m.id}: {m.name} ({m.function}) {m.width}x{m.height}")

   # Single monitor
   m = zm.monitor(1)

``monitors()`` returns ``list[Monitor]``. Results are cached after the first
call; pass ``force_reload=True`` to refresh.

Monitor control
----------------

.. code-block:: python

   m = zm.monitor(1)

   # Alarm control
   m.arm()                              # trigger alarm
   m.disarm()                           # cancel alarm
   status = m.alarm_status()            # check alarm state

   # Update monitor settings
   m.update(Function="Modect", Enabled="1")

   # Daemon status
   m.daemon_status()                    # capture daemon (zmc) status

Streaming & snapshot URLs
--------------------------

.. code-block:: python

   m = zm.monitor(1)

   # MJPEG stream
   url = m.streaming_url()                           # protocol=mjpeg (default)
   url = m.streaming_url(maxfps=5, scale=50)         # with extra params

   # Single-frame snapshot
   url = m.snapshot_url()                            # mode=single
   url = m.snapshot_url(scale=50)

Both URLs are built from the portal URL and ``ZM_PATH_ZMS`` config, with
automatic path deduplication (e.g. ``/zm`` won't be doubled).

PTZ control
------------

.. code-block:: python

   m = zm.monitor("FrontDoor (Video)")

   if m.controllable:
       # Query what the camera can do
       caps = m.ptz_capabilities()
       print(caps.can_move_con, caps.can_zoom, caps.has_presets)

       # Simple commands — defaults to continuous mode
       m.ptz("up")
       m.ptz("stop")
       m.ptz("zoom_in")

       # Move for 2 seconds then auto-stop
       m.ptz("right", stop_after=2.0)

       # Relative or absolute mode
       m.ptz("left", mode="rel")

       # Presets
       m.ptz("home")
       m.ptz("preset", preset=3)

Supported commands: ``up``, ``down``, ``left``, ``right``, ``up_left``,
``up_right``, ``down_left``, ``down_right``, ``zoom_in``, ``zoom_out``,
``stop``, ``home``, ``preset``.

Modes: ``con`` (continuous, default), ``rel`` (relative), ``abs`` (absolute).

Getting zones
--------------

.. code-block:: python

   m = zm.monitor(1)
   zones = m.get_zones()
   for z in zones:
       print(f"{z.name}: {len(z.points)} points, pattern={z.pattern}")

Zones are used by the ML detector for region-based filtering.
