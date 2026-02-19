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
