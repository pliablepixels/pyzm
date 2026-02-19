Detection
==========

.. note::

   This module requires the ``[ml]`` extra: ``pip install "pyzm[ml]"``


Basic usage
------------

Detecting objects in an image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyzm import ZMClient, Detector, StreamConfig

   zm = ZMClient(api_url="https://zm.example.com/zm/api",
                  user="admin", password="secret")
   detector = Detector(models=["yolo11s"])

   m = zm.monitor(1)
   zones = m.get_zones()

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
       ev = zm.event(12345)
       ev.update_notes(result.summary)

``detect_event()`` extracts frames from the event (via the ZM API or
local filesystem), runs each through the detection pipeline, and returns
the best result based on the configured ``frame_strategy``.

``StreamConfig`` controls which frames are extracted, resizing, retry
behaviour, and more.  See `StreamConfig`_ below for details.

Loading from YAML config
~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Architecture overview
----------------------

.. code-block:: text

   Detector
     |
     +-- ModelPipeline
     |     |
     |     +-- YoloBackend        (OpenCV DNN / ONNX)
     |     +-- CoralBackend       (Coral EdgeTPU)
     |     +-- FaceDlibBackend    (dlib / face_recognition)
     |     +-- AlprBackend        (PlateRecognizer / OpenALPR)
     |     +-- RekognitionBackend (AWS Rekognition)
     |     +-- BirdnetBackend     (audio bird species ID)
     |     |
     |     +-- filters (zone, size, pattern, past-detection)
     |
     +-- DetectorConfig
     +-- StreamConfig (for event-based detection)

``Detector`` is the public API. It owns a ``ModelPipeline`` that
sequences the configured backends, applies match strategies, and runs
post-detection filters.

When ``gateway`` is set, ``Detector`` skips local inference and sends
requests to a remote ``pyzm.serve`` server instead.  By default (URL
mode), ``detect_event()`` sends frame URLs and the server fetches
images directly from ZoneMinder.  Set ``gateway_mode="image"`` if the
server cannot reach ZM (see the :doc:`serve guide </guide/serve>`).


Configuration
--------------

DetectorConfig
~~~~~~~~~~~~~~~

Top-level detection settings:

.. code-block:: python

   from pyzm.models.config import DetectorConfig, ModelConfig

   config = DetectorConfig(
       models=[...],                     # list of ModelConfig
       match_strategy="most",            # first | most | most_unique | union
       frame_strategy="most_models",     # first | most | most_unique | most_models
       pattern=".*",                     # global label regex filter
       max_detection_size="50%",         # max bbox size (% of image or "Npx")
       match_past_detections=False,      # compare with previous run
       past_det_max_diff_area="5%",      # area tolerance for past matching
       type_overrides={...},             # per-type overrides (see below)
   )

ModelConfig
~~~~~~~~~~~~

Per-model settings:

.. code-block:: python

   from pyzm.models.config import ModelConfig, ModelFramework, Processor

   model = ModelConfig(
       name="YOLO11s",
       type="object",                    # object | face | alpr
       framework=ModelFramework.OPENCV,  # opencv | coral_edgetpu | face_dlib | ...
       processor=Processor.GPU,          # cpu | gpu | tpu
       weights="/path/to/yolo11s.onnx",
       labels="/path/to/coco.names",
       min_confidence=0.3,
       pattern="(person|car|dog)",
   )

See the API reference for the full list of fields (face-specific,
ALPR-specific, AWS, lock management, etc.).

StreamConfig
~~~~~~~~~~~~~

Controls frame extraction from ZM events:

.. code-block:: python

   from pyzm import StreamConfig

   stream_cfg = StreamConfig(
       frame_set=["snapshot", "alarm", "1"],  # which frames to fetch
       resize=800,               # resize longest edge to N pixels
       max_frames=0,             # 0 = no limit
       start_frame=1,            # first frame index
       frame_skip=1,             # skip every N frames
       max_attempts=1,           # retries on failure
       sleep_between_attempts=3, # seconds between retries
   )

``frame_set`` values: ``"snapshot"`` (the ZM snapshot image), ``"alarm"``
(alarm frames), or integer frame IDs as strings.


Model discovery
----------------

When you pass a string model name to ``Detector(models=["yolo11s"])``,
pyzm resolves it by scanning ``base_path`` (default:
``/var/lib/zmeventnotification/models``):

.. code-block:: text

   /var/lib/zmeventnotification/models/
     yolov4/
       yolov4.weights
       yolov4.cfg
       coco.names
     ultralytics/
       yolo11s.onnx
       yolo26s.onnx
     coral_edgetpu/
       ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
       coco_indexed.names

pyzm resolves names by **directory match** (``yolov4`` →
``yolov4/yolov4.weights``) or **file-stem match** (``yolo11s`` →
``ultralytics/yolo11s.onnx``).

The framework is auto-detected from the file extensions (``.onnx`` =
OpenCV DNN, ``.weights`` + ``.cfg`` = OpenCV Darknet, ``.tflite`` =
Coral EdgeTPU).


The ``ml_sequence`` dict format
--------------------------------

``objectconfig.yml`` uses a nested dict format that maps directly to
``DetectorConfig.from_dict()``:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object,face,alpr,audio"

     object:
       general:
         pattern: "(person|car|dog)"
         same_model_sequence_strategy: "most"
       sequence:
         - name: "YOLO11s"
           object_framework: opencv
           object_weights: /path/to/yolo11s.onnx
           object_labels: /path/to/coco.names
           object_min_confidence: 0.3
           object_processor: cpu

     face:
       general:
         same_model_sequence_strategy: first
       sequence:
         - face_detection_framework: dlib
           known_images_path: /path/to/known_faces
           face_model: cnn

     alpr:
       general:
         same_model_sequence_strategy: first
         pre_existing_labels: ["car", "bus", "truck"]
       sequence:
         - alpr_service: plate_recognizer
           alpr_key: YOUR_KEY

     audio:
       general:
         pattern: ".*"
         same_model_sequence_strategy: first
       sequence:
         - name: BirdNET
           enabled: "yes"
           audio_framework: birdnet
           birdnet_min_conf: 0.5
           birdnet_lat: -1
           birdnet_lon: -1
           birdnet_sensitivity: 1.0
           birdnet_overlap: 0.0

In Python:

.. code-block:: python

   import yaml
   from pyzm import Detector

   with open("objectconfig.yml") as f:
       cfg = yaml.safe_load(f)

   detector = Detector.from_dict(cfg["ml_sequence"])


Supported backends
-------------------

All backends implement the ``MLBackend`` interface (``load()``,
``detect(image)``, ``name``).

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - Framework value
     - Description
   * - YoloBackend
     - ``opencv``
     - OpenCV DNN with Darknet (``.weights``) or ONNX models
   * - CoralBackend
     - ``coral_edgetpu``
     - Google Coral EdgeTPU via ``pycoral``
   * - FaceDlibBackend
     - ``face_dlib``
     - Face recognition using dlib / ``face_recognition``
   * - AlprBackend
     - ``plate_recognizer``, ``openalpr``
     - License plate recognition (cloud or local)
   * - RekognitionBackend
     - ``aws_rekognition``
     - AWS Rekognition API
   * - BirdnetBackend
     - ``birdnet``
     - Audio bird species identification (6500+ species) via ``birdnet-analyzer``


Match and frame strategies
---------------------------

Frame strategies
~~~~~~~~~~~~~~~~~

When multiple frames are extracted from an event, the ``frame_strategy``
determines which frame's detections to return:

- **first** -- return the first frame that has any detections
- **first_new** -- like ``first``, but only counts detections that pass
  past-detection filtering (i.e. genuinely new objects, not parked cars
  that were already detected in a prior run)
- **most** -- return the frame with the most total detections
- **most_unique** -- return the frame with the most unique labels
- **most_models** -- return the frame where the most models contributed
  detections (default)

When two frames tie on the primary metric (same number of detections,
same number of unique labels, or same number of contributing models),
the frame with the higher total confidence sum wins.

Match strategies
~~~~~~~~~~~~~~~~~

When multiple model variants are configured for the same type (e.g. two
object detectors), the ``match_strategy`` determines how their results
are combined:

- **first** -- use results from the first model that detects anything
- **most** -- use the model with the most detections (default)
- **most_unique** -- use the model with the most unique labels
- **union** -- merge all detections from all models

``match_strategy`` can be overridden per model type via
``type_overrides`` (see :ref:`per-type-overrides` below).


Zone-based filtering
---------------------

Zones are polygons that define regions of interest in the camera frame.
Only detections whose bounding box intersects a zone are kept.

.. code-block:: python

   from pyzm.models.zm import Zone

   zones = [
       Zone(name="driveway", points=[(0,300), (600,300), (600,480), (0,480)]),
       Zone(name="porch", points=[(200,0), (400,0), (400,200), (200,200)],
            pattern="person"),
   ]

   result = detector.detect("/path/to/image.jpg", zones=zones)

Each zone can have its own ``pattern`` regex. A detection must match the
zone's pattern *and* physically intersect the polygon to be kept.

Zones also support an ``ignore_pattern`` to suppress specific labels.
When a detection matches a zone's ``ignore_pattern``, it is filtered out
even if it would otherwise match the zone's positive ``pattern``. This is
useful for excluding parked cars or other stationary objects from
specific zones:

.. code-block:: python

   zones = [
       Zone(name="driveway",
            points=[(0,300), (600,300), (600,480), (0,480)],
            pattern="(person|car)",
            ignore_pattern="(car|truck)"),  # suppress parked vehicles
   ]

When using ZoneMinder events, use ``zm.monitor(monitor_id).get_zones()`` to
fetch zones configured in the ZM web UI.


.. _per-type-overrides:

Per-type config overrides
--------------------------

Several settings can be overridden per model type (``object``, ``face``,
``alpr``) via ``type_overrides``.  When a key is set in the per-type
override it takes precedence; otherwise the global value is used.

**Overridable keys:**

- ``match_strategy`` (``same_model_sequence_strategy``)
- ``max_detection_size``
- ``match_past_detections``
- ``past_det_max_diff_area`` (and per-label ``<label>_past_det_max_diff_area``)
- ``ignore_past_detection_labels``
- ``aliases``

**Global only** (a warning is logged if found in a per-type section):

- ``frame_strategy`` — operates above model types (picks best frame
  across all types)
- ``image_path`` — just a directory path, no per-type meaning

In Python:

.. code-block:: python

   from pyzm.models.config import DetectorConfig, TypeOverrides, MatchStrategy, ModelType

   config = DetectorConfig(
       models=[...],
       match_strategy=MatchStrategy.FIRST,        # global default
       match_past_detections=False,                # global default
       type_overrides={
           ModelType.OBJECT: TypeOverrides(
               match_strategy=MatchStrategy.MOST,  # object uses MOST
               match_past_detections=True,          # enabled for object
               past_det_max_diff_area="10%",
           ),
           ModelType.FACE: TypeOverrides(
               match_strategy=MatchStrategy.UNION,  # face uses UNION
           ),
       },
   )

In YAML (``objectconfig.yml``), these keys go in the per-type
``general`` section and ``from_dict()`` populates ``type_overrides``
automatically:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object,face"
       same_model_sequence_strategy: "first"      # global default

     object:
       general:
         same_model_sequence_strategy: "most"      # override for object
         match_past_detections: "yes"
         past_det_max_diff_area: "10%"
         car_past_det_max_diff_area: "15%"
       sequence:
         - ...

     face:
       general:
         same_model_sequence_strategy: "union"     # override for face
       sequence:
         - ...


Past-detection filtering
-------------------------

When ``match_past_detections=True``, pyzm compares detections against a
pickled file from the previous run. Detections whose bounding box is in
roughly the same position (within ``past_det_max_diff_area``) are
filtered out. This prevents repeated notifications for parked cars,
stationary objects, etc.

.. code-block:: python

   config = DetectorConfig(
       models=[...],
       match_past_detections=True,
       past_det_max_diff_area="5%",
       past_det_max_diff_area_labels={"car": "10%"},
       ignore_past_detection_labels=["person"],
       aliases=[["car", "bus", "truck"]],
   )

- ``past_det_max_diff_area`` -- area difference tolerance (default ``"5%"``)
- ``past_det_max_diff_area_labels`` -- per-label overrides
- ``ignore_past_detection_labels`` -- labels to always keep (never filter)
- ``aliases`` -- treat these labels as equivalent when matching

All of these can be overridden per model type using ``type_overrides``
(see :ref:`per-type-overrides`).  Past-detection filtering is applied
per-type: detections are grouped by their ``detection_type`` and each
group uses its own resolved config.  Past data is loaded once, and all
detections are saved once after filtering.


Result objects
---------------

DetectionResult
~~~~~~~~~~~~~~~~

.. code-block:: python

   result.matched          # bool -- any detections?
   result.labels           # list[str]
   result.confidences      # list[float]
   result.boxes            # list[list[int]]  -- [x1,y1,x2,y2] per detection
   result.summary          # str -- "person:97% car:85%"
   result.frame_id         # int | str | None
   result.image            # np.ndarray | None (the source image)
   result.image_dimensions # dict

   result.annotate()       # draw boxes, return annotated image
   result.filter_by_pattern("person")  # new result with only matching labels
   result.to_dict()        # serialize to dict

Detection
~~~~~~~~~~

.. code-block:: python

   det.label           # str
   det.confidence      # float
   det.bbox            # BBox
   det.model_name      # str
   det.detection_type  # str ("object", "face", "alpr", "audio")

BBox
~~~~~

.. code-block:: python

   bbox.x1, bbox.y1   # top-left corner
   bbox.x2, bbox.y2   # bottom-right corner
   bbox.width          # x2 - x1
   bbox.height         # y2 - y1
   bbox.area           # width * height
   bbox.center         # (cx, cy) tuple
