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
