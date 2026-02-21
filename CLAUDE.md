Development notes
-----------------
* `pyzm` is the Python library for ZoneMinder (API, ML pipeline, logging)
* When you update docs, NEVER touch CHANGELOG. It is auto generated
* zmeventnotification is a server ecosystem that uses pyzm extensively. zm_detect.py inside it is located in ~/fiddle/zmeventnotification/hooks
* Any time you make changes to pyzm, always validate that zm_detect.py is up to date and doesn't break
* When bumping up pyzm versions, make sure to modify setup.py in zmeventnotification/hooks to include that version number 
* To run tests: `pytest tests/`
* If you need to access DB, configs etc, access it as `sudo -u www-data`
* Follow DRY principles for coding
* Always write simple code
* Use conventional commit format for all commits:
  * `feat:` new features
  * `fix:` bug fixes
  * `refactor:` code restructuring without behavior change
  * `docs:` documentation only
  * `chore:` maintenance, config, tooling
  * `test:` adding or updating tests
  * Scope is optional: `feat(install):`, `refactor(config):`, etc.
* If you are fixing bugs or creating new features, the process MUST be:
    - Create a GH issue (label it) - ALWAYS create it in pliablepixels/pyzm NEVER Zoneminder/pyzm
    - If developing a feature, create a branch
    - Commit changes referring the issue
    - Wait for the user to confirm before you close the issue
    - Always add test cases, both unit and e2e
    - For e2e do not use ZM_E2E_WRITE - the user will run them manually

Documentation notes
-------------------
- You are an expert document writer and someone who cares deeply that documentation is clear, easy to follow, user friendly and comprehensive and CORRECT.
- Analyze RTD docs and make sure the documents fully represent the capabilities of the system, does not have outdated or incomplete things and is user forward.
- Remember that zm_detect.py leans on pyzm (~/fiddle/zmeventnotification) for most of its functionality. Always validate if what you are doing breaks zm_detect.py and its helpers and make sure you alert me on what to do if you break zm_detect.py
~
