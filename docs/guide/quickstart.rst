Getting Started
===============

This guide walks through the core features of pyzm v2:
connecting to ZoneMinder, running ML detection, and working with results.

Connecting to ZoneMinder
-------------------------

.. code-block:: python

   from pyzm import ZMClient

   zm = ZMClient(
       apiurl="https://zm.example.com/zm/api",
       user="admin",
       password="secret",
       # verify_ssl=False,  # for self-signed certs
   )

   print(f"ZM {zm.zm_version}, API {zm.api_version}")

``apiurl`` must be the full ZM API URL (ending in ``/api``).

Listing monitors
~~~~~~~~~~~~~~~~~

.. code-block:: python

   for m in zm.monitors():
       print(f"Monitor {m.id}: {m.name} ({m.function}) {m.width}x{m.height}")

   # Single monitor
   m = zm.monitor(1)

``monitors()`` returns ``list[Monitor]``. Results are cached after the first
call; pass ``force_reload=True`` to refresh.

Querying events
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   frames = zm.event_frames(event_id=12345)
   for f in frames:
       print(f"Frame {f.frame_id}: type={f.type} score={f.score} delta={f.delta:.2f}s")

   # Find the highest-scoring frame
   best = max(frames, key=lambda f: f.score)
   print(f"Best frame: {best.frame_id} (score={best.score})")

``event_frames()`` returns ``list[Frame]`` with per-frame ``Score``,
``Type`` (Normal/Alarm/Bulk), and ``Delta`` (seconds since event start).

Getting zones
~~~~~~~~~~~~~~

.. code-block:: python

   zones = zm.monitor_zones(monitor_id=1)
   for z in zones:
       print(f"{z.name}: {len(z.points)} points, pattern={z.pattern}")

Zones are used by the ML detector for region-based filtering.


Detecting objects in an image
------------------------------

.. code-block:: python

   from pyzm import Detector

   detector = Detector(models=["yolo11s"])
   result = detector.detect("/path/to/image.jpg")

   if result.matched:
       print(result.summary)  # "person:97% car:85%"

``models`` accepts model name strings (resolved under ``base_path``) or
explicit ``ModelConfig`` objects.

Working with results
~~~~~~~~~~~~~~~~~~~~~

``detect()`` returns a ``DetectionResult``:

.. code-block:: python

   result.matched          # True if any detections
   result.labels           # ["person", "car"]
   result.confidences      # [0.97, 0.85]
   result.boxes            # [[x1,y1,x2,y2], ...]
   result.summary          # "person:97% car:85%"
   result.frame_id         # which frame was selected
   result.image_dimensions # {"original": (h,w), "resized": (rh,rw)}

   # Individual detections
   for det in result.detections:
       print(f"{det.label}: {det.confidence:.0%}")
       print(f"  bbox: ({det.bbox.x1},{det.bbox.y1})-({det.bbox.x2},{det.bbox.y2})")
       print(f"  area: {det.bbox.area}, center: {det.bbox.center}")

   # Annotate and save
   annotated = result.annotate()
   import cv2
   cv2.imwrite("/tmp/detected.jpg", annotated)


Detecting on a ZoneMinder event
---------------------------------

.. code-block:: python

   from pyzm import ZMClient, Detector, StreamConfig

   zm = ZMClient(apiurl="https://zm.example.com/zm/api",
                  user="admin", password="secret")
   detector = Detector(models=["yolo11s"])

   zones = zm.monitor_zones(1)

   result = detector.detect_event(
       zm,
       event_id=12345,
       zones=zones,
       stream_config=StreamConfig(
           frame_set=["snapshot", "alarm", "1"],
           resize=800,
       ),
   )

   if result.matched:
       print(result.summary)
       zm.update_event_notes(12345, result.summary)

``detect_event()`` extracts frames from the event (via the ZM API or
local filesystem), runs each through the detection pipeline, and returns
the best result based on the configured ``frame_strategy``.

``StreamConfig`` controls which frames are extracted, resizing, retry
behaviour, and more. See the :doc:`detection deep-dive </guide/detection>`
for details.


Loading from YAML config
--------------------------

If you have an ``ml_sequence`` dict (from ``objectconfig.yml``):

.. code-block:: python

   ml_options = {
       "general": {
           "model_sequence": "object,face",
       },
       "object": {
           "general": {"pattern": "(person|car)", "same_model_sequence_strategy": "first"},
           "sequence": [
               {
                   "object_framework": "opencv",
                   "object_weights": "/path/to/yolo11s.onnx",
                   "object_labels": "/path/to/coco.names",
                   "object_min_confidence": 0.5,
               },
           ],
       },
   }

   detector = Detector.from_dict(ml_options)
   result = detector.detect("/path/to/image.jpg")

``from_dict()`` parses the nested dict format used by ``objectconfig.yml``
and builds a fully typed ``DetectorConfig`` internally.


Logging
--------

All pyzm internals log to the ``"pyzm"`` stdlib logger.  How you
configure it depends on whether you are using ZoneMinder or not.

Standalone (ML only, no ZoneMinder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only use pyzm for ML detection (no ZM), configure the ``"pyzm"``
logger with stdlib ``logging``:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.DEBUG)          # quick & simple

   # — or — configure just the pyzm logger:
   pyzm_logger = logging.getLogger("pyzm")
   pyzm_logger.setLevel(logging.DEBUG)
   pyzm_logger.addHandler(logging.StreamHandler())   # print to console

   from pyzm import Detector
   detector = Detector(models=["yolo11s"])
   result = detector.detect("/path/to/image.jpg")    # logs appear on console

With ZoneMinder
~~~~~~~~~~~~~~~~

On a ZoneMinder host, ``setup_zm_logging()`` reads ``zm.conf``, the DB
``Config`` table, and environment variables, then writes to ZM's log file,
database, and syslog using the same format as Perl's ``Logger.pm``:

.. code-block:: python

   from pyzm.log import setup_zm_logging

   adapter = setup_zm_logging(name="myapp")
   adapter.Info("Hello from pyzm")
   adapter.Debug(1, "Verbose detail")

   # Override log levels or enable console output
   adapter = setup_zm_logging(name="myapp", override={
       "dump_console": True,
       "log_debug": True,
       "log_level_debug": 5,
   })

``setup_zm_logging()`` returns a :class:`ZMLogAdapter` that provides
``Debug``, ``Info``, ``Warning``, ``Error``, and ``Fatal`` methods
matching the legacy pyzm API.  All pyzm library internals automatically
share the same log handlers via the ``"pyzm"`` logger.
