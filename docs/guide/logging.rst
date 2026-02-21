Logging
========

All pyzm internals log to the ``"pyzm"`` stdlib logger.  How you
configure it depends on whether you are using ZoneMinder or not.

Standalone (ML only, no ZoneMinder)
------------------------------------

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
----------------

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

.. warning::

   ``Fatal()`` logs at CRITICAL level, calls ``close()`` to flush all
   handlers, and then calls ``sys.exit(-1)``.  Only use it for
   unrecoverable errors.

Other ``ZMLogAdapter`` methods:

- ``close()`` -- close and remove all log handlers from the logger
- ``get_config()`` -- return the resolved config dict (useful for debugging
  which settings were applied)

How it works
~~~~~~~~~~~~~

``setup_zm_logging()`` reads configuration from four sources (in order,
later sources override earlier ones):

1. Environment variables (see table below)
2. ZoneMinder config files (``/etc/zm/zm.conf`` and ``conf.d/*.conf``)
3. ZM database ``Config`` table (``ZM_LOG_LEVEL_FILE``, ``ZM_LOG_DEBUG``, etc.)
4. The ``override`` dict you pass directly

Environment variables
^^^^^^^^^^^^^^^^^^^^^^

All environment variables are optional. They are read first and provide
baseline values that can be overridden by ZM config files, the database,
and the ``override`` dict.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Environment variable
     - Config key
     - Description
   * - ``PYZM_CONFPATH``
     - ``conf_path``
     - Path to ZM config directory (default ``/etc/zm``)
   * - ``PYZM_DBUSER``
     - ``dbuser``
     - ZM database username
   * - ``PYZM_DBPASSWORD``
     - ``dbpassword``
     - ZM database password
   * - ``PYZM_DBHOST``
     - ``dbhost``
     - ZM database host (``host`` or ``host:port`` or ``host:/socket``)
   * - ``PYZM_DBNAME``
     - ``dbname``
     - ZM database name
   * - ``PYZM_WEBUSER``
     - ``webuser``
     - Web user for log file ownership (default ``www-data``)
   * - ``PYZM_WEBGROUP``
     - ``webgroup``
     - Web group for log file ownership (default ``www-data``)
   * - ``PYZM_LOGPATH``
     - ``logpath``
     - Log file directory (default ``/var/log/zm``)
   * - ``PYZM_SYSLOGLEVEL``
     - ``log_level_syslog``
     - Syslog handler level (ZM scale: 1=DBG, 0=INF, -1=WAR, -2=ERR, -5=off)
   * - ``PYZM_FILELOGLEVEL``
     - ``log_level_file``
     - File handler level
   * - ``PYZM_DBLOGLEVEL``
     - ``log_level_db``
     - Database handler level
   * - ``PYZM_LOGDEBUG``
     - ``log_debug``
     - Enable debug logging (``1`` = on, ``0`` = off)
   * - ``PYZM_LOGDEBUGLEVEL``
     - ``log_level_debug``
     - Maximum debug sub-level (1--9)
   * - ``PYZM_LOGDEBUGTARGET``
     - ``log_debug_target``
     - Restrict debug logs to matching process names (pipe-separated)
   * - ``PYZM_LOGDEBUGFILE``
     - ``log_debug_file``
     - Override log file path when debug is active
   * - ``PYZM_SERVERID``
     - ``server_id``
     - ZM server ID for database log entries
   * - ``PYZM_DUMPCONSOLE``
     - ``dump_console``
     - Also print to console (``1`` = on)

The ``override`` dict accepts the same keys listed in the "Config key"
column above (e.g. ``override={"dump_console": True, "log_debug": 1,
"log_level_debug": 5}``).  Overrides are applied twice -- before and after
the database read -- so they always take final precedence.

Up to four handlers are attached to the ``"pyzm"`` stdlib logger:

- **File handler** — writes to ``/var/log/zm/<name>.log`` (or the path in
  ``ZM_LOG_DEBUG_FILE``) using ZM's native format matching Perl's ``Logger.pm``
- **Database handler** — writes to ZM's ``Logs`` table via ``mysql.connector``
  (columns: ``TimeKey``, ``Component``, ``ServerId``, ``Pid``, ``Level``,
  ``Code``, ``Message``, ``File``, ``Line``)
- **Syslog handler** — sends to syslog facility ``LOCAL1``
- **Console handler** — enabled when ``dump_console=True``

Each handler's log level is controlled independently by the corresponding
ZM config (``ZM_LOG_LEVEL_FILE``, ``ZM_LOG_LEVEL_DATABASE``,
``ZM_LOG_LEVEL_SYSLOG``).

Signal handlers are registered for log management:
``SIGHUP`` reopens the log file (for log rotation), ``SIGUSR1``/``SIGUSR2``
increase/decrease verbosity at runtime.
