Quick Start
===========

Install pyzm (add ``[ml]`` if you need ML detection):

.. code-block:: bash

   pip install pyzm          # ZM API only
   pip install "pyzm[ml]"    # ZM API + ML detection

Connect to ZoneMinder and list monitors:

.. code-block:: python

   from pyzm import ZMClient

   zm = ZMClient(api_url="https://zm.example.com/zm/api",
                  user="admin", password="secret")

   for m in zm.monitors():
       print(f"{m.name}: {m.function} ({m.width}x{m.height})")

Detect objects in a local image:

.. code-block:: python

   from pyzm import Detector

   detector = Detector(models=["yolo11s"])
   result = detector.detect("/path/to/image.jpg")

   if result.matched:
       print(result.summary)   # "person:97% car:85%"

Next steps
----------

- :doc:`zm_client` -- connecting, authentication, and client options
- :doc:`zm_monitors` -- monitors, zones, PTZ, streaming URLs
- :doc:`zm_events` -- querying events, frames, bulk operations
- :doc:`zm_system` -- system health, configuration, states
- :doc:`logging` -- configuring pyzm logging (standalone and ZM)
- :doc:`detection` -- ML detection pipeline, backends, and config
