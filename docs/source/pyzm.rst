pyzm package
============

ZoneMinder Client
------------------

The typed client for the ZoneMinder REST API. Returns dataclass models
(``Monitor``, ``Event``, ``Zone``) instead of raw dicts.

.. automodule:: pyzm.client
    :members:
    :special-members: __init__
    :undoc-members:

Logging
--------

Logging utilities for pyzm. ``setup_zm_logging()`` provides ZM-native logging
(file, database, syslog) matching Perl's Logger.pm format.

.. automodule:: pyzm.log
    :members: setup_zm_logging, ZMLogAdapter
    :undoc-members:

Machine Learning
------------------

The ML detection pipeline. ``Detector`` is the main entry point --
it manages backends, model sequencing, and result filtering.

.. automodule:: pyzm.ml.detector
    :members:
    :special-members: __init__
    :undoc-members:

.. automodule:: pyzm.ml.pipeline
    :members:
    :special-members: __init__
    :undoc-members:

.. automodule:: pyzm.ml.filters
    :members:
    :undoc-members:

Configuration Models
---------------------

Pydantic v2 models for all pyzm configuration: ZM client settings,
detector/model parameters, and stream extraction options.

.. automodule:: pyzm.models.config
    :members:
    :undoc-members:

.. automodule:: pyzm.models.detection
    :members:
    :undoc-members:

.. automodule:: pyzm.models.zm
    :members:
    :undoc-members:

Remote ML Detection Server
----------------------------

A FastAPI-based server that loads models once and serves detection requests
over HTTP. See the :doc:`serve guide </guide/serve>` for usage.

.. automodule:: pyzm.serve.app
    :members:
    :undoc-members:

.. automodule:: pyzm.serve.auth
    :members:
    :undoc-members:
