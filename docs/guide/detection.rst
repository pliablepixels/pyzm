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


Audio detection (BirdNET)
~~~~~~~~~~~~~~~~~~~~~~~~~~

pyzm supports audio-based detection via the `BirdNET
<https://github.com/kahst/BirdNET-Analyzer>`_ backend, which can identify
6500+ bird species from audio.

**Requirements:**

- ``birdnet-analyzer`` Python package
- ``ffmpeg`` and ``ffprobe`` on ``PATH`` (used to extract audio from video)

.. code-block:: bash

   pip install birdnet-analyzer

**Standalone audio file:**

Use ``detect_audio()`` to analyse an audio file directly, without a
ZoneMinder event:

.. code-block:: python

   from pyzm import Detector
   from pyzm.models.config import DetectorConfig, ModelConfig, ModelType, ModelFramework

   config = DetectorConfig(
       models=[
           ModelConfig(
               name="BirdNET",
               type=ModelType.AUDIO,
               framework=ModelFramework.BIRDNET,
               birdnet_min_conf=0.5,
               birdnet_lat=42.36,    # location for species filtering
               birdnet_lon=-71.06,
           ),
       ],
   )

   detector = Detector(config=config)
   result = detector.detect_audio(
       "/path/to/recording.wav",
       event_week=24,          # ISO week (1-48) for seasonal filtering
       lat=42.36, lon=-71.06,  # fallback if not set in config
   )

   for det in result.detections:
       print(f"{det.label}: {det.confidence:.0%}")
       # e.g. "Northern Cardinal: 87%"

``detect_audio()`` accepts any audio format that ffmpeg can read (WAV,
MP3, MP4, FLAC, etc.).  Audio is split into 3-second chunks and each
chunk is analysed independently; the best confidence per species across
all chunks is returned.

Parameters:

- ``audio_path`` -- path to the audio file
- ``event_week`` -- ISO week number (1--48) for seasonal filtering;
  ``-1`` disables
- ``lat``, ``lon`` -- latitude/longitude for location-based species
  filtering; ``-1`` disables.  These are fallbacks -- values set in
  ``ModelConfig`` (``birdnet_lat``, ``birdnet_lon``) take precedence.

**Event-based audio (automatic):**

When ``detect_event()`` finds an audio-type model in the pipeline, it
automatically extracts audio from the event's video file:

1. Queries the ZM database for the event's ``DefaultVideo`` and the
   monitor's ``Latitude`` / ``Longitude``.
2. Probes the video for an audio stream (via ``ffprobe``).
3. Extracts audio to a temporary WAV file (48 kHz, mono, PCM s16le).
4. Computes the ISO week number from the event's start time.
5. Runs BirdNET analysis alongside the image detection pipeline.
6. Cleans up the temporary WAV file after detection.

No extra code is needed -- just include an audio model in the
``ml_sequence``:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object,audio"
     object:
       sequence:
         - object_framework: opencv
           object_weights: /path/to/yolo11s.onnx
     audio:
       sequence:
         - name: BirdNET
           audio_framework: birdnet
           birdnet_min_conf: 0.5
           birdnet_lat: 42.36
           birdnet_lon: -71.06
           birdnet_sensitivity: 1.0
           birdnet_overlap: 0.0

**BirdNET config parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``birdnet_min_conf``
     - ``0.25``
     - Minimum confidence threshold for species detection
   * - ``birdnet_lat``
     - ``-1.0``
     - Latitude for location-based species filtering (``-1`` = disabled).
       Falls back to the monitor's DB latitude if ``-1``.
   * - ``birdnet_lon``
     - ``-1.0``
     - Longitude for location-based species filtering (``-1`` = disabled).
       Falls back to the monitor's DB longitude if ``-1``.
   * - ``birdnet_sensitivity``
     - ``1.0``
     - Sigmoid sensitivity for score calculation (higher = more sensitive)
   * - ``birdnet_overlap``
     - ``0.0``
     - Overlap between audio chunks (0.0--1.0)

Audio detections have ``detection_type="audio"`` and a dummy bounding box
(``BBox(0, 0, 1, 1)``) since audio has no spatial location.


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
and builds a fully typed ``DetectorConfig`` internally.  It also reads
remote gateway settings from the ``general`` section:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``ml_gateway``
     - URL of a remote ``pyzm.serve`` server (enables remote detection)
   * - ``ml_gateway_mode``
     - ``"url"`` (default) or ``"image"`` (see :doc:`serve guide </guide/serve>`)
   * - ``ml_user``
     - Username for server authentication
   * - ``ml_password``
     - Password for server authentication
   * - ``ml_timeout``
     - HTTP timeout in seconds (default ``60``)

Loading from a YAML config file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``from_config()`` loads a ``Detector`` from a YAML file whose structure
matches ``DetectorConfig`` fields directly (not the ``ml_sequence``
format):

.. code-block:: python

   detector = Detector.from_config("/etc/pyzm/detector.yml")

Example ``detector.yml``:

.. code-block:: yaml

   models:
     - name: "YOLO11s"
       type: object
       framework: opencv
       processor: gpu
       weights: /path/to/yolo11s.onnx
       min_confidence: 0.3
       pattern: "(person|car)"
   match_strategy: most
   frame_strategy: most_models

This is useful when you want to configure detection with Pydantic-style
field names rather than the ``objectconfig.yml`` dict convention.


Architecture overview
----------------------

.. code-block:: text

   Detector
     |
     +-- ModelPipeline
     |     |
     |     +-- YoloOnnx / YoloDarknet  (OpenCV DNN — ONNX or Darknet)
     |     +-- CoralBackend            (Coral EdgeTPU)
     |     +-- FaceDlibBackend         (dlib / face_recognition)
     |     +-- FaceTpuBackend          (Coral TPU face detection)
     |     +-- AlprBackend             (PlateRecognizer / OpenALPR)
     |     +-- RekognitionBackend      (AWS Rekognition)
     |     +-- BirdnetBackend          (audio bird species ID)
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
       type="object",                    # object | face | alpr | audio
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

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``frame_set``
     - ``["snapshot", "alarm", "1"]``
     - Which frames to extract from the event
   * - ``resize``
     - ``800``
     - Resize longest edge to N pixels. ``None`` = no resize.
   * - ``max_frames``
     - ``0``
     - Maximum frames to extract. ``0`` = no limit.
   * - ``start_frame``
     - ``1``
     - First frame index to consider
   * - ``frame_skip``
     - ``1``
     - Process every Nth frame
   * - ``delay``
     - ``0``
     - Delay in seconds before starting frame extraction
   * - ``delay_between_frames``
     - ``0``
     - Delay in seconds between fetching each frame
   * - ``delay_between_snapshots``
     - ``0``
     - Delay in seconds between snapshot requests
   * - ``contig_frames_before_error``
     - ``5``
     - Number of contiguous frame-fetch failures before aborting
   * - ``max_attempts``
     - ``1``
     - Number of retry attempts on failure
   * - ``sleep_between_attempts``
     - ``3``
     - Seconds between retry attempts
   * - ``download``
     - ``False``
     - Download frames to disk instead of keeping in memory
   * - ``download_dir``
     - ``"/tmp"``
     - Directory for downloaded frames (when ``download=True``)
   * - ``disable_ssl_cert_check``
     - ``True``
     - Skip SSL certificate verification when fetching frames
   * - ``save_frames``
     - ``False``
     - Save extracted frames to disk for debugging
   * - ``save_frames_dir``
     - ``"/tmp"``
     - Directory for saved frames
   * - ``delete_after_analyze``
     - ``False``
     - Delete downloaded frames after detection completes
   * - ``convert_snapshot_to_fid``
     - ``False``
     - Convert ``"snapshot"`` to an actual frame ID

In YAML (``stream_sequence`` dict), ``from_dict()`` accepts string
conventions: ``resize: "no"`` → ``None``, ``resize: "800"`` → ``800``,
``frame_set: "snapshot,alarm,1"`` → list, ``"yes"``/``"no"`` → bool.


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
   * - YoloOnnx
     - ``opencv`` (``.onnx``)
     - ONNX/Ultralytics YOLO models via OpenCV DNN (letterboxing, end-to-end support)
   * - YoloDarknet
     - ``opencv`` (``.weights``)
     - Legacy Darknet YOLO models (``.weights`` + ``.cfg``) via OpenCV DNN
   * - CoralBackend
     - ``coral_edgetpu``
     - Google Coral EdgeTPU object detection via ``pycoral``
   * - FaceDlibBackend
     - ``face_dlib``
     - Face recognition using dlib / ``face_recognition`` with KNN matching
   * - FaceTpuBackend
     - ``face_tpu``
     - Coral TPU face detection (detection only, no recognition)
   * - AlprBackend
     - ``plate_recognizer``, ``openalpr``
     - License plate recognition (PlateRecognizer cloud, OpenALPR cloud, or OpenALPR command-line)
   * - RekognitionBackend
     - ``aws_rekognition``
     - AWS Rekognition API
   * - BirdnetBackend
     - ``birdnet``
     - Audio bird species identification (6500+ species) via ``birdnet-analyzer``

The YOLO backend is selected automatically based on the weights file extension:
``.onnx`` files use ``YoloOnnx`` (with letterboxing, end-to-end model support,
and 640px default input), while ``.weights`` files use ``YoloDarknet`` (with
416px default input). This dispatch is handled by the ``create_yolo_backend()``
factory — you just set ``framework: opencv`` and the right backend is chosen.


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

.. note::

   When no zones are passed (or the list is empty), zone filtering is
   skipped entirely and all detections pass through.

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


Pre-existing label gates
-------------------------

``pre_existing_labels`` lets you gate a model so it only runs when a
previous model in the pipeline has already detected one of the listed
labels. This is most commonly used for ALPR: there is no point running
license-plate recognition unless an object detector has already found a
car, bus, or truck.

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object,alpr"
     object:
       sequence:
         - object_framework: opencv
           object_weights: /path/to/yolo11s.onnx
     alpr:
       general:
         pre_existing_labels: ["car", "bus", "truck"]
       sequence:
         - alpr_service: plate_recognizer
           alpr_key: YOUR_KEY

In Python:

.. code-block:: python

   ModelConfig(
       name="PlateRecognizer",
       type=ModelType.ALPR,
       framework=ModelFramework.PLATE_RECOGNIZER,
       alpr_key="YOUR_KEY",
       pre_existing_labels=["car", "bus", "truck"],
   )

If no model in a prior type has detected any of the listed labels, the
model is skipped entirely (logged at debug level). ``pre_existing_labels``
can be set per-model or in the per-type ``general`` section.


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

The pickle file is stored at ``<image_path>/past_detections.pkl``.
``image_path`` defaults to ``"/tmp"``
and can be set in ``DetectorConfig`` or via ``objectconfig.yml``
(which typically sets it to ``"/var/lib/zmeventnotification/images"``).


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
   result.error_boxes      # list[BBox] -- detections filtered out by zone/pattern/size

   result.annotate()       # draw boxes on image (error_boxes drawn in red)
   result.filter_by_pattern("person")  # new result with only matching labels
   result.to_dict()        # serialize to dict

   # Reconstruct from a wire dict (e.g. JSON from pyzm.serve)
   result = DetectionResult.from_dict(data)

``error_boxes`` contains bounding boxes for objects that were detected by
the model but subsequently filtered out (wrong zone, wrong pattern, too
large, etc.). These are drawn in red (unfilled) by ``annotate()`` so you
can see what was rejected.

Detection
~~~~~~~~~~

.. code-block:: python

   det.label           # str
   det.confidence      # float
   det.bbox            # BBox
   det.model_name      # str
   det.detection_type  # str ("object", "face", "alpr", "audio")

   det.matches_pattern("(person|car)")  # bool -- regex match against label

BBox
~~~~~

.. code-block:: python

   bbox.x1, bbox.y1   # top-left corner
   bbox.x2, bbox.y2   # bottom-right corner
   bbox.width          # x2 - x1
   bbox.height         # y2 - y1
   bbox.area           # width * height
   bbox.center         # (cx, cy) tuple

   bbox.as_list()             # [x1, y1, x2, y2]
   bbox.as_polygon_coords()   # [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] for Shapely
