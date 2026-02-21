System & Config
================

System health
--------------

.. code-block:: python

   zm.is_running()          # True if ZM daemon is running
   zm.system_load()         # {"1min": 0.5, "5min": 0.3, "15min": 0.2}
   zm.disk_usage()          # disk percent info from ZM
   zm.timezone()            # e.g. "America/New_York"

Configuration
--------------

.. code-block:: python

   # List all config parameters
   all_configs = zm.configs()

   # Get a single config by name
   cfg = zm.config("ZM_LANG_DEFAULT")
   print(cfg["Value"])

   # Set a config value (system configs are read-only)
   zm.set_config("ZM_LANG_DEFAULT", "en_us")

Shared memory (real-time monitor status)
-----------------------------------------

``pyzm.zm.shm.SharedMemory`` reads ZoneMinder's memory-mapped files
(``/dev/shm/zm.mmap.<monitor_id>``) for real-time monitor state without
API calls. This is the same shared memory that ZM's internal processes use.

Supports ZM 1.36.x, 1.36.34+, and 1.38+ struct layouts (auto-detected).

.. code-block:: python

   from pyzm.zm.shm import SharedMemory

   # Context manager ensures cleanup
   with SharedMemory(monitor_id=1) as shm:
       print(shm.is_valid())        # True if mmap can be read
       print(shm.is_alarmed())      # True if currently in ALARM state
       print(shm.alarm_state())     # {"id": 0, "state": "STATE_IDLE"}
       print(shm.last_event())      # last event ID (int)
       print(shm.cause())           # {"alarm_cause": "...", "trigger_cause": "..."}
       print(shm.trigger())         # trigger text, showtext, cause, state

       # Raw data dicts
       sd = shm.get_shared_data()   # all SharedData fields
       td = shm.get_trigger_data()  # all TriggerData fields

       # Full dump (both shared + trigger)
       data = shm.get()

SharedData fields include: ``state``, ``capture_fps``, ``analysis_fps``,
``last_event``, ``alarm_x``, ``alarm_y``, ``valid``, ``signal``,
``imagesize``, ``last_frame_score``, ``startup_time``, ``heartbeat_time``,
``alarm_cause``, and more (varies by ZM version).

.. note::

   SharedMemory requires read access to ``/dev/shm/zm.mmap.*`` files.
   On a typical ZM installation, run as ``www-data`` or with appropriate
   permissions.

States, servers, and storage
-----------------------------

.. code-block:: python

   # List ZM states
   for s in zm.states():
       print(s["Name"])

   # State control
   zm.stop()                # stop ZM
   zm.start()               # start ZM
   zm.restart()             # restart ZM
   zm.set_state("my_state") # switch to a named state

   # Multi-server setups
   zm.servers()             # list all ZM servers
   zm.storage()             # list storage areas with disk usage
