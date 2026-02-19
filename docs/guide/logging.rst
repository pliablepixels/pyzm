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
