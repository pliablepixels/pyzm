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

Constructor parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``api_url``
     - (required)
     - Full ZM API URL (e.g. ``https://server/zm/api``)
   * - ``user``
     - ``None``
     - ZM username. ``None`` when auth is disabled.
   * - ``password``
     - ``None``
     - ZM password
   * - ``portal_url``
     - auto
     - Full portal URL (e.g. ``https://server/zm``). Auto-derived from
       ``api_url`` when not provided.
   * - ``verify_ssl``
     - ``True``
     - Set to ``False`` for self-signed certificates
   * - ``timeout``
     - ``30``
     - HTTP request timeout in seconds
   * - ``db_user``
     - ``None``
     - Override the ZM database username (normally read from ``zm.conf``)
   * - ``db_password``
     - ``None``
     - Override the ZM database password
   * - ``db_host``
     - ``None``
     - Override the ZM database host (e.g. ``"dbhost"`` or ``"dbhost:3307"``)
   * - ``db_name``
     - ``None``
     - Override the ZM database name
   * - ``conf_path``
     - ``None``
     - Path to the ZM config directory (default ``/etc/zm``). Useful when
       ``zm.conf`` lives in a non-standard location.
   * - ``config``
     - ``None``
     - A pre-built ``ZMClientConfig``. When provided, all other keyword
       args are ignored.

Authentication
---------------

pyzm handles authentication internally.  When ``user`` and ``password``
are provided, it obtains an API token and refreshes it automatically.
Legacy (non-token) authentication is also supported for older ZM
installations.

Set ``verify_ssl=False`` if your ZM server uses a self-signed TLS
certificate.

Database access
----------------

Some operations — ``ev.tag()``, ``ev.path()``, and audio extraction for
BirdNET — require a direct MySQL connection to the ZM database.  By
default, pyzm reads credentials from ``/etc/zm/zm.conf`` (the same file
ZoneMinder uses).

If the user running pyzm cannot read ``zm.conf`` (e.g. permission
denied), you can pass database credentials explicitly:

.. code-block:: python

   zm = ZMClient(
       api_url="https://zm.example.com/zm/api",
       user="admin",
       password="secret",
       db_user="zmuser",
       db_password="zmpass",
       db_host="localhost",
   )

   ev = zm.event(12345)
   ev.tag(["person"])   # uses the explicit DB credentials
   path = ev.path()     # same

The merge strategy is:

1. Try to read ``zm.conf`` (or ``conf_path`` if set).
2. Overlay any explicit ``db_*`` parameters — explicit values always win.
3. Fall back to ``"localhost"`` for host and ``"zm"`` for database name.

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
