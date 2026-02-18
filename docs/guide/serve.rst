Remote ML Detection Server
===========================

``pyzm.serve`` is a built-in FastAPI server that loads ML models once
and serves detection requests over HTTP. This lets you offload GPU-heavy
inference to a dedicated machine.

.. code-block:: text

   URL mode (default)                     GPU box
   +-----------------+   frame URLs      +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |  fetch from ZM   |
   |   gateway=...)  | <---------------- |  detect & return |
   +-----------------+  DetectionResult  +------------------+
                                                |
                                         +------v-----------+
                                         | ZoneMinder API   |
                                         +------------------+

   Image mode (gateway_mode="image")      GPU box
   +-----------------+     HTTP/JPEG     +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |   YOLO11 (GPU)  |
   |   gateway=...,  | <---------------- |   Coral TPU      |
   |   gateway_mode= |  DetectionResult  +------------------+
   |     "image")    |
   +-----------------+

Two detection modes are available:

- **URL mode** (default) -- the client sends frame URLs to the
  ``/detect_urls`` endpoint and the *server* fetches images directly from
  ZoneMinder.  This avoids transferring every frame through the client.
- **Image mode** -- the client fetches frames from ZM, JPEG-encodes them,
  and uploads each one to the ``/detect`` endpoint.  Use this when the
  server cannot reach ZoneMinder directly.


Deployment scenarios
---------------------

Scenario 1: ZM + Event Server + hooks + pyzm (same box)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything runs on the same machine. The ZoneMinder Event Server (ES)
triggers hook scripts which call ``zm_detect.py``, and detection runs
locally via the ``Detector`` class.

.. code-block:: text

   ZoneMinder --> zmeventnotification.pl (ES)
                     |
                     v
                  zm_event_start.sh
                     |
                     v
                  zm_detect.py --> Detector (local GPU/CPU)

**objectconfig.yml** (no ``remote`` section needed):

.. code-block:: yaml

   ml:
     ml_sequence:
       general:
         model_sequence: "object"
       object:
         general:
           pattern: "(person|car)"
         sequence:
           - name: YOLO11s
             object_weights: "/var/lib/zmeventnotification/models/ultralytics/yolo11s.onnx"
             object_labels: "/var/lib/zmeventnotification/models/yolov4/coco.names"
             object_framework: opencv
             object_processor: gpu

**Test locally:**

.. code-block:: bash

   sudo -u www-data python3 -m pyzm.zm_detect \
       --config /etc/zm/objectconfig.yml \
       --eventid 12345 \
       --debug


Scenario 2: ZM + hooks + pyzm (same box, no ES)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as Scenario 1 but without the Event Server. ZoneMinder calls
``zm_detect.py`` directly via its ``EventStartCommand`` / ``EventEndCommand``
recording settings.

.. code-block:: text

   ZoneMinder EventStartCommand --> zm_detect.py --> Detector (local)

**ZoneMinder Console -> Click on Monitor Source -> Recording:**

.. code-block:: text

   EventStartCommand = /usr/bin/python3 -m pyzm.zm_detect -c /etc/zm/objectconfig.yml -e %EID% -m %MID% -r "%EC%" -n

**objectconfig.yml** is the same as Scenario 1.


Scenario 3: ZM box + remote GPU box (split architecture)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detection runs on a separate GPU machine. The ZM box runs ``zm_detect.py``
which sends requests to the remote ``pyzm.serve`` server over HTTP.

.. code-block:: text

   ZM box                              GPU box
   +-------------------+               +------------------------+
   | zm_detect.py      |   HTTP        | pyzm.serve             |
   | Detector(         | ------------> |   --models all         |
   |   gateway=...)    |               |   --processor gpu      |
   |                   | <------------ |   --port 5000          |
   +-------------------+  JSON result  +------------------------+

**GPU box setup:**

.. code-block:: bash

   pip install pyzm[serve]
   python -m pyzm.serve --models all --processor gpu --port 5000

Or with specific models and auth:

.. code-block:: bash

   python -m pyzm.serve \
       --models yolo11s yolo26s \
       --processor gpu \
       --port 5000 \
       --auth --auth-user admin --auth-password secret \
       --token-secret my-jwt-secret

**ZM box objectconfig.yml:**

.. code-block:: yaml

   ml:
     ml_sequence:
       general:
         model_sequence: "object"
         ml_gateway: "http://192.168.1.100:5000"
         # ml_gateway_mode: "image"   # uncomment if server can't reach ZM directly
         ml_fallback_local: "yes"
       object:
         general:
           pattern: "(person|car)"
         sequence:
           - object_framework: opencv
             object_weights: "/var/lib/zmeventnotification/models/ultralytics/yolo11s.onnx"

By default, URL mode is used -- the server fetches frames directly from ZM.
Set ``ml_gateway_mode: "image"`` if the server cannot reach ZoneMinder
(the client will JPEG-encode and upload frames instead).


Available models
-----------------

Model names and discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

Model names passed via ``--models`` (or ``Detector(models=[...])``) are
resolved against ``--base-path`` on disk. There are no hardcoded presets --
any name you pass is looked up as follows:

1. **Directory match** -- ``base_path/<name>/`` containing a weight file
2. **File stem match** -- any ``<name>.onnx``, ``<name>.weights``, or
   ``<name>.tflite`` in any subdirectory of ``base_path``

The framework is inferred from the file extension:

- ``.onnx`` -- OpenCV DNN (ONNX runtime)
- ``.weights`` -- OpenCV DNN (Darknet format, also needs a ``.cfg`` file)
- ``.tflite`` -- Coral Edge TPU runtime (processor forced to ``tpu``)

Label files are auto-detected from the same directory (``.names``,
``.txt``, ``.labels``). For Darknet models, ``.cfg`` files are also
discovered automatically.

The ``--processor`` flag (``cpu``, ``gpu``, ``tpu``) applies to all
discovered models (except ``.tflite`` which always uses ``tpu``).


``--models all`` (lazy loading)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you pass ``--models all`` to the server, **every** model in
``--base-path`` is discovered and registered, but weights are **not**
loaded into memory at startup. Instead, each backend loads its weights
on the first request that uses it.

This is useful when you have many models but don't want to consume
GPU memory for all of them upfront.

.. code-block:: bash

   python -m pyzm.serve --models all --base-path /data/models --processor gpu

Use the ``GET /models`` endpoint to check which models are available
and whether their weights have been loaded.


Model directory layout
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   /var/lib/zmeventnotification/models/
   +-- yolov4/
   |   +-- yolov4.weights
   |   +-- yolov4.cfg
   |   +-- coco.names
   +-- ultralytics/
   |   +-- yolo11s.onnx
   |   +-- yolo11n.onnx
   |   +-- yolo26s.onnx
   +-- coral_edgetpu/
   |   +-- ssd_mobilenet_v2.tflite
   |   +-- coco_labels.txt


Server setup
-------------

Installation
~~~~~~~~~~~~~

The ``[serve]`` extra automatically includes all ML dependencies.

.. code-block:: bash

   pip install "pyzm[serve]"


CLI options
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--host``
     - ``0.0.0.0``
     - Bind address
   * - ``--port``
     - ``5000``
     - Bind port
   * - ``--models``
     - ``yolo11s``
     - Model names (space-separated). Use ``all`` to auto-discover every
       model in ``--base-path`` (loaded lazily on first request).
   * - ``--base-path``
     - ``/var/lib/zmeventnotification/models``
     - Directory containing model subdirectories
   * - ``--processor``
     - ``cpu``
     - ``cpu``, ``gpu``, or ``tpu``
   * - ``--auth``
     - off
     - Enable JWT authentication
   * - ``--auth-user``
     - ``admin``
     - Username (when auth enabled)
   * - ``--auth-password``
     - (empty)
     - Password (when auth enabled)
   * - ``--token-secret``
     - ``change-me``
     - Secret key used to sign JWT tokens. **Change this in production.**
   * - ``--config``
     - (none)
     - Path to a YAML config file (``ServerConfig``). Overrides CLI flags.


YAML config file
~~~~~~~~~~~~~~~~~

Instead of CLI flags, you can provide a YAML config file via ``--config``:

.. code-block:: bash

   python -m pyzm.serve --config /etc/pyzm/serve.yml

Example ``serve.yml``:

.. code-block:: yaml

   host: "0.0.0.0"
   port: 5000
   models:
     - yolo11s
     - yolo26s
   base_path: "/var/lib/zmeventnotification/models"
   processor: gpu
   auth_enabled: true
   auth_username: admin
   auth_password: "my-secret-password"
   token_secret: "a-strong-random-secret"
   token_expiry_seconds: 3600

All fields correspond to ``ServerConfig`` attributes. When using
``--config``, CLI flags are ignored.


Client usage
-------------

Using the ``Detector`` API:

.. code-block:: python

   from pyzm import Detector

   # URL mode (default) -- server fetches frames from ZM
   detector = Detector(models=["yolo11s"], gateway="http://gpu-box:5000")

   # Image mode -- client uploads JPEG-encoded frames
   # detector = Detector(models=["yolo11s"], gateway="http://gpu-box:5000",
   #                     gateway_mode="image")

   # With authentication:
   # detector = Detector(models=["yolo11s"], gateway="http://gpu-box:5000",
   #                     gateway_username="admin", gateway_password="secret")

   # detect() always uploads the image (single-image mode)
   result = detector.detect("/path/to/image.jpg")
   print(result.summary)

   # detect_event() uses URL mode by default -- sends frame URLs,
   # server fetches them directly from ZM
   result = detector.detect_event(zm_client, event_id=12345,
                                   stream_config=stream_cfg)

URL mode only applies to ``detect_event()`` calls.  Single-image
``detect()`` calls always upload the image regardless of this setting.

Using ``from_dict()``
~~~~~~~~~~~~~~~~~~~~~~

The ``ml_gateway`` key in the ``general`` section of ``ml_options``
automatically enables remote mode:

.. code-block:: python

   ml_options = {
       "general": {
           "model_sequence": "object",
           "ml_gateway": "http://gpu-box:5000",
           # "ml_gateway_mode": "image",  # uncomment if server can't reach ZM
           # "ml_user": "admin",
           # "ml_password": "secret",
       },
       "object": {
           "general": {"pattern": ".*"},
           "sequence": [...],
       },
   }

   detector = Detector.from_dict(ml_options)
   result = detector.detect(image)


Authentication
---------------

When the server is started with ``--auth``, clients must first obtain a
JWT token via ``/login``, then pass it as a Bearer token on subsequent
requests. The ``Detector`` gateway mode handles this automatically.

The ``--token-secret`` flag controls the secret key used to sign JWT
tokens. **Always set this to a strong random value in production.**
The default (``change-me``) is insecure.

Manual flow:

.. code-block:: bash

   # Login
   TOKEN=$(curl -s -X POST http://gpu-box:5000/login \
       -H 'Content-Type: application/json' \
       -d '{"username":"admin","password":"secret"}' \
       | jq -r .access_token)

   # Detect
   curl -X POST http://gpu-box:5000/detect \
       -H "Authorization: Bearer $TOKEN" \
       -F file=@/path/to/image.jpg

Tokens expire after ``--token-expiry`` seconds (default 3600).


API reference
--------------

``GET /health``
~~~~~~~~~~~~~~~~

Health check. Returns:

.. code-block:: json

   {"status": "ok", "models_loaded": true}

``GET /models``
~~~~~~~~~~~~~~~~

Returns the list of available models and their load status. Useful with
``--models all`` to check which backends have been lazily loaded.

.. code-block:: json

   {
     "models": [
       {"name": "yolo11s", "type": "object", "framework": "opencv", "loaded": true},
       {"name": "yolo26s", "type": "object", "framework": "opencv", "loaded": false}
     ]
   }

``POST /detect``
~~~~~~~~~~~~~~~~~

Run detection on an uploaded image (image mode).

- **Content-Type:** ``multipart/form-data``
- **Parameters:**
  - ``file`` (required) -- JPEG/PNG image
  - ``zones`` (optional) -- JSON string of zone list
- **Auth:** Bearer token (when auth enabled)
- **Returns:** ``DetectionResult`` as JSON (image field excluded)

``POST /detect_urls``
~~~~~~~~~~~~~~~~~~~~~~

Run detection on images fetched from URLs (URL mode).

- **Content-Type:** ``application/json``
- **Body:**

  .. code-block:: json

     {
       "urls": [
         {"frame_id": "snapshot", "url": "https://zm.example.com/zm/index.php?view=image&eid=123&fid=snapshot"},
         {"frame_id": "1", "url": "https://zm.example.com/zm/index.php?view=image&eid=123&fid=1"}
       ],
       "zm_auth": "token=abc123...",
       "zones": [{"name": "driveway", "value": [[0,0],[100,0],[100,100],[0,100]]}],
       "verify_ssl": false
     }

- **Auth:** Bearer token (when auth enabled)
- **Returns:** ``DetectionResult`` as JSON (best frame selected by ``frame_strategy``)

The server appends ``zm_auth`` to each URL and fetches the image via HTTP GET.
Frame strategy (``first``, ``most``, ``most_unique``, ``most_models``)
is applied server-side to pick the best result.

``POST /login``
~~~~~~~~~~~~~~~~

Obtain a JWT token (only available when ``--auth`` is enabled).

- **Content-Type:** ``application/json``
- **Body:** ``{"username": "...", "password": "..."}``
- **Returns:** ``{"access_token": "...", "expires": 3600}``


objectconfig.yml remote section
---------------------------------

In a ZoneMinder event notification setup, configure the remote gateway
in ``objectconfig.yml``:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object"
       ml_gateway: "http://gpu-box:5000"
       # ml_gateway_mode: "image"        # uncomment if server can't reach ZM
       ml_user: "admin"
       ml_password: "secret"
       ml_fallback_local: yes

     object:
       general:
         pattern: "(person|car)"
       sequence:
         - object_framework: opencv
           object_weights: /path/to/yolo11s.onnx
           object_labels: /path/to/coco.names

When ``ml_gateway`` is set, detection requests are sent to the remote
server.  URL mode is the default -- the server fetches frames directly
from ZM.  Set ``ml_gateway_mode: "image"`` if the server cannot reach
ZoneMinder.  If ``ml_fallback_local`` is ``yes`` and the remote server
is unreachable, detection falls back to local inference.
