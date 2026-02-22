# Changelog

All notable changes to this project will be documented in this file.


## [2.1.5] - 2026-02-22

### Bug Fixes

- fall back to default Storage row when ZM_DIR_EVENTS missing ([cf4d442](https://github.com/pliablepixels/pyzm/commit/cf4d4427010659c57df23ddfb1b3076ca6d354dc))
- handle StorageId=0 in _event_path by falling back to ZM_DIR_EVENTS ([03e71cb](https://github.com/pliablepixels/pyzm/commit/03e71cbed3e84359ccadb70862571f7b4b3bf253))

## [2.1.4] - 2026-02-22

### Bug Fixes

- downgrade zm.conf warning to debug when explicit DB creds provided ([b6e0a97](https://github.com/pliablepixels/pyzm/commit/b6e0a978ef6a19a5837bd7e8b18f878f708a8306))

### Documentation

- update CHANGELOG for v2.1.4 ([9103b6b](https://github.com/pliablepixels/pyzm/commit/9103b6bb8f4084fdf61a6d970f9f3c34c5f903f4))
- expand face recognition install with BLAS and reinstall steps ([0a1594e](https://github.com/pliablepixels/pyzm/commit/0a1594e2a9342fb2e2c9310a09214b32ec9830e7))
- took out some arch notes ([539b4ec](https://github.com/pliablepixels/pyzm/commit/539b4ec20669d317ddeb8529c592266ff242179b))
- comprehensive accuracy and completeness audit of all guide pages ([785c65c](https://github.com/pliablepixels/pyzm/commit/785c65cf78c160bad08dcfc58cd15ee30fda9055))
- move raw() section from events page to client page ([3ab58b4](https://github.com/pliablepixels/pyzm/commit/3ab58b447b8a04e042ebf0e2659e8c8ee006fb85))
- add upgrading section for PyPI and source installs ([0d843c1](https://github.com/pliablepixels/pyzm/commit/0d843c13750ae34c790fa1cf23636881027c2c0b))

### Features

- pass DB credentials through ZMClientConfig ([7324a65](https://github.com/pliablepixels/pyzm/commit/7324a651474da32218643796227b9662b64a7248))

### Miscellaneous

- ver bump ([5f9ee66](https://github.com/pliablepixels/pyzm/commit/5f9ee66a60f8ae98af37f8ac1a5b4bcb11b2ddda))

### Refactoring

- merge legacy ML implementations with backend adapters ([#23](https://github.com/pliablepixels/pyzm/issues/23)) ([583525a](https://github.com/pliablepixels/pyzm/commit/583525a2f0757507d105901470203e2bc9cb4a3e))

## [2.1.3] - 2026-02-20

### Bug Fixes

- auto-detect movement mode from control profile capabilities ([55778ca](https://github.com/pliablepixels/pyzm/commit/55778ca9656094f389b2706db4aeefed015d99b9))
- read version from file to avoid pydantic import on RTD ([a559941](https://github.com/pliablepixels/pyzm/commit/a5599411b60924c4c93f3452eeb01f3ac40ad9d2))

### Documentation

- update CHANGELOG for v2.1.3 ([58bb8fd](https://github.com/pliablepixels/pyzm/commit/58bb8fdf34771fa1d9b7c2f6c7cca0535092751e))
- restructure sidebar into Getting Started, ZM API, and ML sections ([f5665d1](https://github.com/pliablepixels/pyzm/commit/f5665d19e68fee42f41edb0003579cd57dd5cc49))

### Miscellaneous

- ver bump ([a4a1c8f](https://github.com/pliablepixels/pyzm/commit/a4a1c8f764a8d27d2af05a1f8a5d7122fa2e6b6d))

## [2.1.2] - 2026-02-19

### Bug Fixes

- read version dynamically from pyzm.__version__ in conf.py ([7d349a0](https://github.com/pliablepixels/pyzm/commit/7d349a0e00af8d1e23b9d31000a06c232d6c9e1d))

### Documentation

- update CHANGELOG for v2.1.2 ([0bb5f2e](https://github.com/pliablepixels/pyzm/commit/0bb5f2eef12da0c00c791ca6069c6ee129a545e3))
- added documentation notes and es reliance ([771a931](https://github.com/pliablepixels/pyzm/commit/771a931cbc1cc2a62f666988c34a57adc583072e))
- fix inaccuracies and gaps across documentation ([8af47cc](https://github.com/pliablepixels/pyzm/commit/8af47cc74c4ab13dfbbc699f3c7cba3fe62dfa62))

### Features

- add raw() method to all ZM API model objects ([ba77844](https://github.com/pliablepixels/pyzm/commit/ba77844a3a20915fca6050636257fe6558dc8229))

### Miscellaneous

- bump version to v2.1.2 ([a752740](https://github.com/pliablepixels/pyzm/commit/a7527402ee5f13f36b98283c97823d5a1b76ebfc))

## [2.1.1] - 2026-02-18

### Documentation

- update CHANGELOG for v2.1.1 ([0fd7981](https://github.com/pliablepixels/pyzm/commit/0fd7981d5238f01b2a03be6bb44ff26d97873ac9))
- fix remote.py docstring to show URL mode as default ([312e667](https://github.com/pliablepixels/pyzm/commit/312e667726d8a81636e8af036b0bf2d289a6b758))
- add RTD links to all examples ([3e7e455](https://github.com/pliablepixels/pyzm/commit/3e7e455d2edf1f812d1c4e0e5137f2052a6a037d))
- simplify examples, add URL mode to remote.py ([3110ec1](https://github.com/pliablepixels/pyzm/commit/3110ec17de45bbb5d939f39b782200ff657d4208))

### Features

- accept /login when auth disabled, add --debug flag ([cff3151](https://github.com/pliablepixels/pyzm/commit/cff3151133cefb3e11ec7dc5feeeed9b90474bf7))
- change default gateway_mode from "image" to "url" ([8d2b675](https://github.com/pliablepixels/pyzm/commit/8d2b67591b1faf7e0cd4da88ac9cedf010c713e0))

### Miscellaneous

- bump version to v2.1.1 ([05fac76](https://github.com/pliablepixels/pyzm/commit/05fac761af125ee858bacc394a7b4d1c12017b53))

## [2.1.0] - 2026-02-18

### Bug Fixes

- download model weights to project_dir, not CWD ([19aef17](https://github.com/pliablepixels/pyzm/commit/19aef179420267a30a26449a90b08f89015552d0))
- use single file uploader to avoid duplicate uploads ([ee77049](https://github.com/pliablepixels/pyzm/commit/ee770493ad5f4123c197a825f677243d7fca2167))
- clarify why images are needed for regrouping ([4247c55](https://github.com/pliablepixels/pyzm/commit/4247c559f1761b8036da2776266ab767699612c4))
- explain auto-detect purpose and hide when not useful ([be796f7](https://github.com/pliablepixels/pyzm/commit/be796f7b2bd895c9be40472ac52cb587a26efd57))
- fix project load rerun loop and improve layout alignment ([a406802](https://github.com/pliablepixels/pyzm/commit/a406802127c6b107ea191184c4d82bbe1946a38a))
- use media file manager for canvas background images ([8e9b7f2](https://github.com/pliablepixels/pyzm/commit/8e9b7f20f9063c0cd76306ff0dbf1c7e2408c50d))
- shim image_to_url for streamlit-drawable-canvas compat ([e360eb2](https://github.com/pliablepixels/pyzm/commit/e360eb2d7c8d5475b5f12f277ea1efa0dd8e712b))
- fix upload rerun loop and show next steps after upload ([7e6a8a0](https://github.com/pliablepixels/pyzm/commit/7e6a8a00387f896c59ecb4f5287da8a344c3707e))
- use native Streamlit dark theme instead of injected CSS ([ce1659f](https://github.com/pliablepixels/pyzm/commit/ce1659fef77ec73ad8afffe6cc0862e336b3bf50))
- scan individual model files, auto-populate classes from ONNX ([900399d](https://github.com/pliablepixels/pyzm/commit/900399d2a30a3f5233de77229759cf1558088c94))
- use single model for auto-detect, redesign UI flow ([de5e9b8](https://github.com/pliablepixels/pyzm/commit/de5e9b834412c00770ad7f747d3fa6663974f338))

### Documentation

- update CHANGELOG for v2.1.0 ([1e755fd](https://github.com/pliablepixels/pyzm/commit/1e755fd1e2963855d45c9992a689920f4a3ce097))
- document all new ZMClient API methods in quickstart ([78e2b7e](https://github.com/pliablepixels/pyzm/commit/78e2b7e617475dead1f52e1801058b50dff65542))
- add standalone logging guide, fix stale url= references ([82450d9](https://github.com/pliablepixels/pyzm/commit/82450d97a4f5c3f1343f080fa3742b2bed492eb6))
- add BirdNET to detection guide and feature list ([7186b58](https://github.com/pliablepixels/pyzm/commit/7186b5831669988291e1905264c7615320666208))
- update birdnet and optional deps for venv install paths ([fa81958](https://github.com/pliablepixels/pyzm/commit/fa81958fab5981ee8ca2d0cec1402e4522b9e213))
- update RTD logo to match new logo ([cd64bdb](https://github.com/pliablepixels/pyzm/commit/cd64bdb3fd05401bd545986a7b43e3b4bcd1da17))
- add Coral EdgeTPU deprecation note with pycoral[#149](https://github.com/pliablepixels/pyzm/issues/149) link ([5022a73](https://github.com/pliablepixels/pyzm/commit/5022a7319227124d7f8b5948959524981f5b68e4))
- add training guide and update install instructions for extras ([f820cee](https://github.com/pliablepixels/pyzm/commit/f820cee8bf14dbfdbd8423d24c1c9c89cd84e713))
- link zmNg to ReadTheDocs instead of GitHub ([0ecced5](https://github.com/pliablepixels/pyzm/commit/0ecced54a289d54222da458fa89c4983982d20eb))

### Features

- add OOP resource methods on Monitor/Event and PTZ control ([f2ea43c](https://github.com/pliablepixels/pyzm/commit/f2ea43ca8f0ce58229b16431dd9d3bba826abb36))
- add missing ZM API features to ZMClient ([5f88385](https://github.com/pliablepixels/pyzm/commit/5f883852bfa2889bb4a6619a30bd93e4afb842fc))
- add Detector.detect_audio() and BirdNET examples ([ed77522](https://github.com/pliablepixels/pyzm/commit/ed775229b24eb708f8f207873f31062da603d6bb))
- add BirdNET audio bird recognition backend ([14771da](https://github.com/pliablepixels/pyzm/commit/14771daaf9527acea2249f64eff87e53d0f1e256))
- use shared venv instead of --break-system-packages ([05645cd](https://github.com/pliablepixels/pyzm/commit/05645cda7c60c441faae3da61c767a3003ead98f))
- add headless CLI training pipeline ([d6df249](https://github.com/pliablepixels/pyzm/commit/d6df249bf51b512093b47af1a77e99823192da8f))
- UI polish, MPS support, folder picker, and docs ([cf7b9a3](https://github.com/pliablepixels/pyzm/commit/cf7b9a39cd37b0017121b20537601cf29f7a9539))
- grid review UI, recursive ONNX model scan, and local import ([ab54b78](https://github.com/pliablepixels/pyzm/commit/ab54b78bbab5afe88dfda6822e5dec8a52134847))
- remove Upload phase, add date pickers, and UX improvements ([d32ddd0](https://github.com/pliablepixels/pyzm/commit/d32ddd0356d3d3e82a74b4d0f15e54e5e54b6859))
- verification-based review workflow and UX improvements ([fb2e349](https://github.com/pliablepixels/pyzm/commit/fb2e349789700e2240fe180e750c35d7bfdc3ab1))
- skip training for remap-only projects ([b2e5815](https://github.com/pliablepixels/pyzm/commit/b2e58155f28c1418cbbe8f702d18f438b5619adc))
- labeled annotations, color-coded boxes, min 2 images ([808e304](https://github.com/pliablepixels/pyzm/commit/808e3047d1d0418e4bec69f7186e64cb4e5780a9))
- add class grouping and dark UI theme ([8bac331](https://github.com/pliablepixels/pyzm/commit/8bac331dcb2b495d4515e13f2891181d7276826e))
- add YOLO fine-tuning training module (pyzm[train]) ([b47fbab](https://github.com/pliablepixels/pyzm/commit/b47fbab08353c4d7d8111871e18ab7af0cc84e1e))

### Miscellaneous

- switch setup.py to find_packages(), clean up install docs ([4521063](https://github.com/pliablepixels/pyzm/commit/45210631ee7a01d4b3e25af6dceb90a1a00f2eed))
- gitignore YOLO .pt weight files ([dc7ac09](https://github.com/pliablepixels/pyzm/commit/dc7ac09dd16a6a4140e7902414e0d957ff5eb0c9))
- new logo ([764653f](https://github.com/pliablepixels/pyzm/commit/764653f708777fbf63a186a9d21aa2c1fb42f81e))
- remove balloons animation from training complete ([84077f1](https://github.com/pliablepixels/pyzm/commit/84077f12bd6824ab55149fadb07787f3a8f2562c))
- consolidate logo, add RTD logo, slim docs dependencies ([4062f09](https://github.com/pliablepixels/pyzm/commit/4062f09afa4190c16a61db5aecf62e350ac5c93f))
- replace deprecated use_container_width with width ([a57e490](https://github.com/pliablepixels/pyzm/commit/a57e490ce1097b9ca8640d805e63762455fa5c32))

### Refactoring

- remove all pre-v2 legacy code ([3a2633e](https://github.com/pliablepixels/pyzm/commit/3a2633ede1863f6911eb69e54832d8a6bdac5f49))
- rename ZMClient url= to apiurl=, remove auto-append ([f8acc72](https://github.com/pliablepixels/pyzm/commit/f8acc7210a5a39da34449f28835673ea8a1b5a6e))
- adaptive upload+label flow based on class types ([24f2a1e](https://github.com/pliablepixels/pyzm/commit/24f2a1eed6758916d1185223ff18f38c7958b724))
- merge auto-detect into label step, simplify flow ([c77e7e1](https://github.com/pliablepixels/pyzm/commit/c77e7e185164bbabf4286fad3b4509e07da34ef0))

### revert

- remove label_map remap-only shortcut ([8724836](https://github.com/pliablepixels/pyzm/commit/8724836eae2a393f3dacfeda36904858b7661f0f))

## [2.0.7] - 2026-02-15

### Documentation

- update CHANGELOG for v2.0.7 ([1566872](https://github.com/pliablepixels/pyzm/commit/1566872faf73e50f7f5e93e43254ccdb7a1d4016))
- update YOLO model references to match actual disk layout ([599097f](https://github.com/pliablepixels/pyzm/commit/599097fe5cd05e295cf9910efc1e11a66355aa8d))
- fix incorrect defaults and YAML keys in detection/serve guides ([57710ce](https://github.com/pliablepixels/pyzm/commit/57710cefe29c78ec6ad4da82bd7ed0dae6cff6d9))

### Miscellaneous

- bump version to v2.0.7 ([a7bf0f3](https://github.com/pliablepixels/pyzm/commit/a7bf0f3a45f8524458fd5d7f665f4dce2f0154be))

## [2.0.6] - 2026-02-15

### Bug Fixes

- detect near-zero garbled e2e output in YOLO26 ONNX models ([3e39906](https://github.com/pliablepixels/pyzm/commit/3e39906185c5425c7193cfe6028811aedc307862))

### Documentation

- update CHANGELOG for v2.0.6 ([8d7ffa8](https://github.com/pliablepixels/pyzm/commit/8d7ffa8490ab481eb39786b743e240c83693d4f3))
- clarify that CHANGELOG is not to be touched ([f898ffc](https://github.com/pliablepixels/pyzm/commit/f898ffc02db4dd96f2b07a7d2c506c43320d9104))
- add per-type overrides section and remove stale "legacy" wording ([a72da12](https://github.com/pliablepixels/pyzm/commit/a72da12b1fb3893e8e8ffcbd6dafc0f4229f8215))
- add --pyzm-debug to EventStartCommand example ([86a8481](https://github.com/pliablepixels/pyzm/commit/86a84819c5df20e05516a37c66bedc35178b28be))

### Features

- per-type config overrides and processor logging ([0a41321](https://github.com/pliablepixels/pyzm/commit/0a4132100ddc5e7787fdd150acf54ca09f518e88))

### Miscellaneous

- bump version to v2.0.6 ([8973b29](https://github.com/pliablepixels/pyzm/commit/8973b294f7595eb53106c33522289181581b1f59))
- clarified this is a rewrite ([d045258](https://github.com/pliablepixels/pyzm/commit/d045258516e09929641a286f57c5c4dfa8338125))

### Refactoring

- unify logger hierarchy, remove ZMLog and setup_logging ([bb4ca52](https://github.com/pliablepixels/pyzm/commit/bb4ca527b545df04f129e4deb4d59be56de2142a))

## [2.0.5] - 2026-02-15

### Bug Fixes

- add confidence tiebreak and FIRST_NEW strategy ([c0973be](https://github.com/pliablepixels/pyzm/commit/c0973be1d4edf4cab016a4082d20d25bdcaef8c2))

### Documentation

- update CHANGELOG for v2.0.5 ([677f1a7](https://github.com/pliablepixels/pyzm/commit/677f1a72b908cc90787061d9ad3eb299b0e5a866))
- update guides for upstream fixes ([1996662](https://github.com/pliablepixels/pyzm/commit/199666217a6366f1bf1a9cf87afe40fe8f77a9a2))

### Features

- add version bump option when tag already exists ([94bf651](https://github.com/pliablepixels/pyzm/commit/94bf651bf45f310994d00b6b851a7f578f0d6839))
- add ZM-native logging via setup_zm_logging(), drop SQLAlchemy ([ee6d739](https://github.com/pliablepixels/pyzm/commit/ee6d7394368c836239f864a639616b94cdd28041))
- add session-level lock interface for EdgeTPU ([4d59d82](https://github.com/pliablepixels/pyzm/commit/4d59d82dfbb6afcd2cf704e4a35571af05aadc7e))
- support ignore_pattern in zone filtering ([5c66d3f](https://github.com/pliablepixels/pyzm/commit/5c66d3fcf087fecad1eb2981dd2d706f4310e96d))
- add event_frames() for per-frame metadata ([f52d358](https://github.com/pliablepixels/pyzm/commit/f52d35811a61d06cc99ce417a514d9076d202471))
- add analysis_fps, bandwidth to MonitorStatus ([607d720](https://github.com/pliablepixels/pyzm/commit/607d72094e261adf8fce5b82290e361456b16bc1))

### Miscellaneous

- bump version to v2.0.5 ([56003ad](https://github.com/pliablepixels/pyzm/commit/56003ad933f7845ccd1c43174e9ce60be86c0baa))

## [2.0.4] - 2026-02-14

### Bug Fixes

- add ZM 1.36.34 SharedData struct layout (768 bytes) ([600768f](https://github.com/pliablepixels/pyzm/commit/600768f6c360cc0649b624116e961d0d7126ee20))

### Documentation

- update CHANGELOG for v2.0.4 ([ab3eb45](https://github.com/pliablepixels/pyzm/commit/ab3eb45165817c57a70d78f4d493986b5cc8ff78))

### Features

- add ZoneMinder live E2E tests and fix events filter bug ([26d1ecd](https://github.com/pliablepixels/pyzm/commit/26d1ecd01b8348ffe266db83254e1ec620374480))

### Miscellaneous

- remove unused make_package.sh ([b6d2218](https://github.com/pliablepixels/pyzm/commit/b6d2218a6181629c438cd1da60bda89117b023ce))
- ver bump ([b52bc2e](https://github.com/pliablepixels/pyzm/commit/b52bc2e85c2d195f2f9609c713117e2413f95533))

## [2.0.3] - 2026-02-14

### Bug Fixes

- skip commits starting with https in changelog ([8436c5c](https://github.com/pliablepixels/pyzm/commit/8436c5ce08a24df5cf37c8f0e5d91031c1ca37c3))
- add native YOLO26 end-to-end output parsing with pre-NMS fallback ([575247b](https://github.com/pliablepixels/pyzm/commit/575247b98cfd1c795f4f0fa938a51b37a998ef5f))
- letterbox preprocessing and xyxy coord format for ONNX YOLO models ([96198d5](https://github.com/pliablepixels/pyzm/commit/96198d5670d042ec6081205f4fe1415883cfe347))
- eagerly load backends in _ensure_pipeline() for serve ([445b745](https://github.com/pliablepixels/pyzm/commit/445b74507918051291e3c425db7a9a59c3196050))
- add sphinx_rtd_theme to docs requirements and _static dir ([5ac9bf2](https://github.com/pliablepixels/pyzm/commit/5ac9bf20a7e4f2eb2781b36e024affae15c0b90d))
- restore pyzm.ml.alpr and pyzm.ml.face in setup.py ([260f666](https://github.com/pliablepixels/pyzm/commit/260f66679db1e2ba96aad68ac47c9e26983d9e9b))
- handle non-UTF-8 bytes when decoding shared memory strings ([4ba0ee5](https://github.com/pliablepixels/pyzm/commit/4ba0ee502d7a2ab9644b7d869020df32d8eab5c4))
- don't skip file/console logging when DB reflection fails ([de16b5d](https://github.com/pliablepixels/pyzm/commit/de16b5daed8f34475da74fa039b3669278ce8a89))
- strip whitespace from stream parameter to handle non-breaking spaces ([3874a26](https://github.com/pliablepixels/pyzm/commit/3874a26fce03d3837062d33949810bf7edf08636))
- warn when OpenCV version is too old for ONNX YOLOv26 models ([a913280](https://github.com/pliablepixels/pyzm/commit/a913280643d0480bd5a88af5b2ae15eec35cf329))
- apply defaults for unset env vars and handle missing zm.conf gracefully ([47a8cea](https://github.com/pliablepixels/pyzm/commit/47a8ceaf506a7f08b5da5446bebaab85931329c2))
- fall back to CPU when CUDA backend is unavailable for YOLO inference ([0223dd1](https://github.com/pliablepixels/pyzm/commit/0223dd1482c3fed0a89da28804bd082fe0c8c3e9))
- prefix all log messages with model name for clarity ([84e6574](https://github.com/pliablepixels/pyzm/commit/84e6574c4d045466591e296178a8c0d457465121))
- expose lock_name on wrapper classes for dedup logic ([8db190a](https://github.com/pliablepixels/pyzm/commit/8db190a41d10c69931b95e5efa3fb2f2d59abc5a))
- prevent deadlock when multiple models share the same GPU lock ([2a84e90](https://github.com/pliablepixels/pyzm/commit/2a84e90531ac60cb2f12f52b3b3ef90758088a35))
- Later versions of OpenCV have changed format for layers ([271d111](https://github.com/pliablepixels/pyzm/commit/271d111c02b208e53b8cd06aeba60862c2508067))

### Documentation

- update CHANGELOG for v2.0.3 ([9be2ade](https://github.com/pliablepixels/pyzm/commit/9be2ade821241a9151e1d3f00cbe1544c7caa4c1))
- update CHANGELOG for v2.0.3 ([8762414](https://github.com/pliablepixels/pyzm/commit/8762414c7c0de2ee63494820728e133fdd7534e5))
- add developer release notes and --skip-pypi flag ([c726bce](https://github.com/pliablepixels/pyzm/commit/c726bcefdf6b4c40cbe61bc5b04375c1f3b573c7))
- expand sidebar navigation to show subsection indicators ([09c4948](https://github.com/pliablepixels/pyzm/commit/09c494816440ac0eb531dc751e862b0db5103672))
- specify OpenCV 4.13+ requirement for ONNX YOLO models ([407422d](https://github.com/pliablepixels/pyzm/commit/407422d73d429ebe962c6cd53053193caa8e9451))
- link README installation section to RTD guide ([65f67dd](https://github.com/pliablepixels/pyzm/commit/65f67dd563378b7306581aab9f8a3f128fd02c67))
- add installation guide with --break-system-packages note ([b4f15d9](https://github.com/pliablepixels/pyzm/commit/b4f15d9c13e549b388d4fa928df7af5fb8fb99f2))
- fix ES link to point to v7+ RTD ([5e095c6](https://github.com/pliablepixels/pyzm/commit/5e095c6e425f0261720b9e20f658cc38bd6aa661))
- add ES v7+ and zmNg to sidebar ([6bba6e9](https://github.com/pliablepixels/pyzm/commit/6bba6e92c447f745ec686960920a79e8fa33dd97))
- auto-update copyright year in Sphinx config ([c958458](https://github.com/pliablepixels/pyzm/commit/c95845808b1d8815ced9f2d50520074c6946451f))
- add related projects section with ES, zmNg, and zmNinja links ([6e9cdc7](https://github.com/pliablepixels/pyzm/commit/6e9cdc7f1fa65728de2e71fab831370d6d6dd87c))
- add testing guide to README and RTD ([a3706cf](https://github.com/pliablepixels/pyzm/commit/a3706cf039c0d432343c3ed4f63fc5b2bf5fc59b))
- fix EventStartCommand location in serve.rst ([1577fe2](https://github.com/pliablepixels/pyzm/commit/1577fe22bbc67a85330d8b173e76bb1aee657a4c))
- point README to pyzmv2.readthedocs.io ([87ecbf1](https://github.com/pliablepixels/pyzm/commit/87ecbf13e2ad2e76ef74b49fce6de12ea6aefbaf))
- fix ReadTheDocs build for pyzm.serve autodoc ([a6ab684](https://github.com/pliablepixels/pyzm/commit/a6ab684d3ce3b76e18252f89a383cbaa16331d40))
- clarified never to hit ZM ([6a54eae](https://github.com/pliablepixels/pyzm/commit/6a54eae62b706bc6be96321145547f8097543ce4))

### Features

- add yolo11n, yolo11s, yolo26n, yolo26s model presets ([2843619](https://github.com/pliablepixels/pyzm/commit/2843619436c2b74326c97ef150d5d207a31b9a31))
- add --models all lazy loading and overhaul serve docs ([4596f63](https://github.com/pliablepixels/pyzm/commit/4596f6339c8e489a41db867e8aed65b75c7af5a8))
- add URL-mode remote detection for pyzm.serve ([297149d](https://github.com/pliablepixels/pyzm/commit/297149dfaa618dfeb56e729cbe0b64e24e243bd6))
- add pyzm.serve remote ML detection server and Detector gateway mode ([8ee09bf](https://github.com/pliablepixels/pyzm/commit/8ee09bff22d9e862b1bc21f0f97aabfde3f06ac8))
- wire up missing config fields and past-detection filtering ([5948937](https://github.com/pliablepixels/pyzm/commit/5948937cdcb9d91dca6cc5c39aa49118972db1fd))
- add ZMClient.event_path() and fix setup.py module list ([c3e362c](https://github.com/pliablepixels/pyzm/commit/c3e362cd86192d1c1ade24cc27ed4f8a31b51359))
- pyzm v2 rewrite — Pydantic models, new ML pipeline, ZM API client ([b28a7a0](https://github.com/pliablepixels/pyzm/commit/b28a7a0c86254509829116ecab470a6c8cce200c))
- add ZM 1.38 SharedData struct support with size-based auto-detection ([7ee4e0b](https://github.com/pliablepixels/pyzm/commit/7ee4e0b55b37b7a57be7b487c8d62aa11ae9b360))
- add ONNX model support to Yolo class via OpenCV DNN ([8608afe](https://github.com/pliablepixels/pyzm/commit/8608afe49b79a3a3e86c09cfb95d7e68bd9c3032))

### Miscellaneous

- add release tooling (git-cliff + make_release.sh) and fix CLAUDE.md ([1bb0aa9](https://github.com/pliablepixels/pyzm/commit/1bb0aa96c9cb34be73349baded2ecafd5f044025))
- update package URLs, modernize build, bump to 2.0.1 ([957384a](https://github.com/pliablepixels/pyzm/commit/957384a475ad9f8b9a2d29944bff2e98a70d3e48))
- bump version to 0.4.1 ([1facaf1](https://github.com/pliablepixels/pyzm/commit/1facaf1dec2cc96facb3d967d8c067c523837d39))
- bump version to 0.4.0 ([d36120a](https://github.com/pliablepixels/pyzm/commit/d36120a4766a0675ea5f7f9a09ac6e82c9c2da06))
- CLAUDE.md first version ([7a0e5f8](https://github.com/pliablepixels/pyzm/commit/7a0e5f8f354884a59928d464d7d4982c716b3301))
- ver bump ([f16a30d](https://github.com/pliablepixels/pyzm/commit/f16a30db13c13fcdea24c787f7a3fe492d2300b8))

### Refactoring

- remove model presets, resolve all names from disk ([95e7fe2](https://github.com/pliablepixels/pyzm/commit/95e7fe237a270152dfad947714698d8a256c5bf1))
- update template_fill to use ${key} syntax instead of legacy {{key}} ([e034833](https://github.com/pliablepixels/pyzm/commit/e03483331d682de48fb2adce7fed850d44b6e796))
- switch read_config/get from ConfigParser to YAML ([43e9ef7](https://github.com/pliablepixels/pyzm/commit/43e9ef78c55ce8546bdff52744c54b725bd9481a))
- split Yolo into YoloDarknet and YoloOnnx subclasses ([a8f60ec](https://github.com/pliablepixels/pyzm/commit/a8f60ec57c28925f4b3931848847d326ae30d1a8))
- remove direct ultralytics backend, use ONNX via OpenCV DNN ([07de62b](https://github.com/pliablepixels/pyzm/commit/07de62b8e33df0c3a630ea2946cad1815d837a00))

### Testing

- add 89 e2e tests covering every objectconfig feature ([13b578d](https://github.com/pliablepixels/pyzm/commit/13b578dca4430d15db3ad7a7b0d31b5318bd2147))
<!-- generated by git-cliff -->
