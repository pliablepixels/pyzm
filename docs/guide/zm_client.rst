Client & Authentication
=======================

Connecting to ZoneMinder
-------------------------

.. code-block:: python

   from pyzm import ZMClient

   zm = ZMClient(
       api_url="https://zm.example.com/zm/api",
       user="admin",
       password="secret",
       # verify_ssl=False,  # for self-signed certs
   )

   print(f"ZM {zm.zm_version}, API {zm.api_version}")

``api_url`` must be the full ZM API URL (ending in ``/api``).

Authentication
---------------

pyzm handles authentication internally.  When ``user`` and ``password``
are provided, it obtains an API token and refreshes it automatically.
Legacy (non-token) authentication is also supported for older ZM
installations.

Set ``verify_ssl=False`` if your ZM server uses a self-signed TLS
certificate.

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
