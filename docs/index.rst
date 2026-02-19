pyzm -- Python for ZoneMinder
==============================

**pyzm** is a Python library for interacting with the
`ZoneMinder <https://www.zoneminder.com/>`_ surveillance system.
It provides:

- A typed client for the ZoneMinder REST API (monitors, events, states)
- An ML detection pipeline supporting YOLO, Coral EdgeTPU, face recognition, ALPR, and audio recognition (BirdNET)
- A remote ML detection server (``pyzm.serve``) for offloading GPU work
- A web UI for fine-tuning YOLO models on your own data (``pyzm.train``)
- Pydantic v2 configuration models and typed detection results

`Source on GitHub <https://github.com/pliablepixels/pyzm>`__

See :doc:`guide/installation` for detailed instructions, including notes on
system-managed Python environments.

Quick example
--------------

.. code-block:: python

   from pyzm import ZMClient, Detector

   # Connect to ZoneMinder
   zm = ZMClient(api_url="https://zm.example.com/zm/api",
                  user="admin", password="secret")

   for m in zm.monitors():
       print(f"{m.name}: {m.function} ({m.width}x{m.height})")

   # Detect objects in a local image
   detector = Detector(models=["yolo11s"])
   result = detector.detect("/path/to/image.jpg")

   if result.matched:
       print(result.summary)       # "person:97% car:85%"
       result.annotate()           # draw bounding boxes on the image

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guide/installation
   guide/quickstart

.. toctree::
   :maxdepth: 2
   :caption: ZoneMinder API

   guide/zm_client
   guide/zm_monitors
   guide/zm_events
   guide/zm_system
   guide/logging

.. toctree::
   :maxdepth: 2
   :caption: Machine Learning

   guide/detection
   guide/serve
   guide/training

.. toctree::
   :maxdepth: 2
   :caption: Testing

   guide/testing

.. toctree::
   :maxdepth: 2
   :caption: Examples

   example

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   source/modules

.. toctree::
   :hidden:
   :caption: Related Projects

   Event Server v7+ <https://zmeventnotificationv7.readthedocs.io/en/latest/>
   zmNg <https://zmng.readthedocs.io/>


Related Projects
==================

`Event Notification Server v7+ <https://zmeventnotificationv7.readthedocs.io/en/latest/>`__
        Push notifications, WebSockets, and MQTT for ZoneMinder events
`zmNg Documentation <https://zmng.readthedocs.io/>`__
        The newer-generation app for ZoneMinder
`zmNinja Documentation <https://zmninja.readthedocs.io/en/latest/index.html>`__
        Documentation for zmNinja

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
