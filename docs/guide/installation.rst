Installation
============

.. important::

   pyzm and zmeventnotification now install into a **shared Python virtual
   environment** (``/opt/zoneminder/venv`` by default) instead of installing
   globally with ``pip install --break-system-packages``.

   Why the change:

   - Modern Linux distributions (Debian 12+, Ubuntu 23.04+, Fedora 38+)
     mark the system Python as *externally managed* (PEP 668) and actively
     block global pip installs.
   - ``--break-system-packages`` bypasses that protection but can break
     OS tools that depend on the system Python.
   - Multiple ZoneMinder components (pyzm, zmeventnotification hooks) need
     to share a single Python environment — a dedicated venv gives them
     isolation from the OS while still sharing packages with each other.

Requirements
------------

- Python 3.10 or newer
- Pydantic >= 2.0
- OpenCV 4.13+ (``cv2``) -- required for ML detection features (ONNX YOLO models need 4.13+ for full operator support)
- A running ZoneMinder instance (for API features)

Path A: Install from PyPI
--------------------------

The simplest path — no need to clone the repo.

**1. Create the venv** (skip if it already exists, e.g. from zmeventnotification's
installer):

.. code-block:: bash

   sudo python3 -m venv /opt/zoneminder/venv --system-site-packages
   sudo /opt/zoneminder/venv/bin/pip install --upgrade pip setuptools wheel
   sudo chown -R www-data:www-data /opt/zoneminder/venv

**2. Install pyzm:**

.. code-block:: bash

   # Everything (ZM + ML + detection server + training UI):
   sudo /opt/zoneminder/venv/bin/pip install "pyzm[full]"

   # ML detection + remote server:
   sudo /opt/zoneminder/venv/bin/pip install "pyzm[ml,serve]"

   # ML detection + training UI:
   sudo /opt/zoneminder/venv/bin/pip install "pyzm[ml,train]"

   # ML detection only:
   sudo /opt/zoneminder/venv/bin/pip install "pyzm[ml]"

   # ZM API client only (monitors, events, zones — no ML):
   sudo /opt/zoneminder/venv/bin/pip install pyzm

Path B: Install from source
-----------------------------

Clone the repo first, then use the helper script which handles venv creation,
OpenCV shim, and ownership in one step.

.. code-block:: bash

   git clone https://github.com/pliablepixels/pyzm.git
   cd pyzm

   # Everything (ZM + ML + detection server + training UI):
   sudo ./scripts/setup_venv.sh --extras full

   # ML detection + remote server:
   sudo ./scripts/setup_venv.sh --extras ml,serve

   # ML detection only:
   sudo ./scripts/setup_venv.sh --extras ml

   # ZM API client only (monitors, events, zones — no ML):
   sudo ./scripts/setup_venv.sh

   # Custom venv path:
   sudo ZM_VENV=/usr/local/zm/venv ./scripts/setup_venv.sh

If the venv already exists (e.g. created by zmeventnotification's installer),
the script reuses it and just installs/upgrades pyzm.

**Quick editable install** — if you are making local changes to the pyzm source
and want to test them immediately without re-running the full setup script:

.. code-block:: bash

   sudo /opt/zoneminder/venv/bin/pip install -e .

This installs pyzm in *editable* (development) mode so the venv picks up your
source changes directly.

The script will:

1. Install ``python3-venv`` if it is missing (Debian/Ubuntu/Fedora/CentOS).
2. Create the venv with ``--system-site-packages`` so distro-packaged
   libraries (e.g. OpenCV built from source) are still visible.
3. Register an ``opencv-python`` shim if a source/system OpenCV is already
   importable, preventing pip from overwriting it when installing packages
   like ``ultralytics`` that unconditionally depend on ``opencv-python``.
4. Install pyzm (and any requested extras) into the venv.
5. Set ownership to ``www-data`` (configurable via ``ZM_VENV_OWNER``).

What each extra installs
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Extra
     - Additional packages
   * - *(base)*
     - requests, pydantic, dateparser, mysql-connector-python, python-dotenv
   * - ``[ml]``
     - numpy, Pillow, onnx, Shapely, portalocker
   * - ``[serve]``
     - ``[ml]`` deps + fastapi, uvicorn, python-multipart, PyJWT, PyYAML
   * - ``[train]``
     - ``[ml]`` deps + ultralytics, streamlit, streamlit-drawable-canvas, st-clickable-images, PyYAML
   * - ``[full]``
     - All of the above (``[ml]`` + ``[serve]`` + ``[train]``)

.. note::

   The ``[train]`` extra pulls in ``ultralytics``, which unconditionally
   requires ``opencv-python`` from PyPI — even if you already have OpenCV
   installed from source or your system package manager.

   The install script (``scripts/setup_venv.sh`` and zmeventnotification's
   ``install.sh``) attempts to work around this by creating a compatibility
   shim: if ``cv2`` is already importable when the venv is created, a fake
   ``opencv-python`` dist-info entry is registered so that pip considers the
   requirement satisfied and skips the download.

   If your source-built OpenCV is nevertheless overwritten, you can fix it
   manually:

   .. code-block:: bash

      /opt/zoneminder/venv/bin/pip uninstall opencv-python opencv-python-headless
      # Reinstall or rebuild your custom OpenCV — see below

Building OpenCV from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pip ``opencv-python`` package is CPU-only. For GPU-accelerated inference
you need to build OpenCV from source:

- `Apple Silicon (macOS) <https://gist.github.com/pliablepixels/d0605aab085592e1d3b6bb9033bdc835>`_
- `Ubuntu 24.04 with CUDA <https://gist.github.com/pliablepixels/73d61e28060c8d418f9fcfb1e912e425>`_

The venv is created with ``--system-site-packages``, so a system-wide OpenCV
built from source is automatically visible inside the venv.

macOS (development)
-------------------

On macOS the venv can live anywhere — there is no ``www-data`` user:

.. code-block:: bash

   python3 -m venv ~/zm-venv --system-site-packages
   ~/zm-venv/bin/pip install -e ".[ml,serve,train]"

Upgrading
---------

**PyPI install:**

.. code-block:: bash

   # Replace [full] with whichever extras you originally installed:
   sudo /opt/zoneminder/venv/bin/pip install --upgrade "pyzm[full]"

**Source install:**

.. code-block:: bash

   cd pyzm
   git pull
   sudo ./scripts/setup_venv.sh --extras full   # same extras you used before

Optional dependencies
---------------------

Some ML backends have additional requirements that are **not** installed
automatically:

- **BirdNET audio recognition** -- ``birdnet-analyzer`` for identifying bird species from audio in ZM events
- (DEPRECATED) **Coral EdgeTPU** -- ``pycoral`` and the Edge TPU runtime
- **Face recognition (dlib)** -- ``dlib``, ``face_recognition``
- **ALPR** -- a running OpenALPR or Plate Recognizer service
- **AWS Rekognition** -- ``boto3`` with configured AWS credentials

Install these into the venv as needed for your detection pipeline:

.. code-block:: bash

   /opt/zoneminder/venv/bin/pip install birdnet-analyzer  # BirdNET audio

**Face recognition (dlib)** — takes a while and installs a lot of dependencies,
which is why it is not included automatically:

.. code-block:: bash

   sudo apt-get install libopenblas-dev liblapack-dev libblas-dev  # not mandatory, but gives a good speed boost
   /opt/zoneminder/venv/bin/pip install face_recognition            # installs dlib automatically

.. note::

   If you use Google Coral, you can do face *detection* (not recognition) via
   the Coral libraries and can skip this section.

If you installed ``face_recognition`` earlier **without** the BLAS libraries,
reinstall both ``dlib`` and ``face_recognition`` so dlib is built with OpenBLAS
support:

.. code-block:: bash

   /opt/zoneminder/venv/bin/pip uninstall dlib face-recognition
   sudo apt-get install libopenblas-dev liblapack-dev libblas-dev   # the important part
   /opt/zoneminder/venv/bin/pip install dlib --verbose --no-cache-dir  # make sure it finds openblas
   /opt/zoneminder/venv/bin/pip install face_recognition

.. note::

   Google has discontinued the Coral EdgeTPU product line and the
   ``pycoral`` library is no longer maintained. Installing it on modern
   Python (3.10+) and recent Linux distributions requires manual
   workarounds. See `pycoral#149
   <https://github.com/google-coral/pycoral/issues/149>`_ for
   community discussion and installation tips.

Verifying the installation
--------------------------

.. code-block:: bash

   # Base install (ZM API client):
   /opt/zoneminder/venv/bin/python -c "import pyzm; print(pyzm.__version__)"

   # ML detection (requires pyzm[ml]):
   /opt/zoneminder/venv/bin/python -c "from pyzm import Detector; print('ML OK')"

To verify the training UI is available:

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -c "from pyzm.train import check_dependencies; check_dependencies(); print('OK')"
