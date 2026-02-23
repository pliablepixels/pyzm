Model Fine-Tuning
=================

pyzm includes tools for fine-tuning YOLO object detection models on your own
data. This lets ZoneMinder users teach the model to recognize custom objects
(e.g. specific vehicles, pets, packages) or improve detection accuracy for
objects the base model struggles with.

Two modes are available:

- **Web UI** -- a guided Streamlit app for importing images, reviewing/correcting
  detections, and training. Best for interactive workflows.
- **Headless CLI** -- a single command that runs the full pipeline
  (validate → import → split → train → export) without a browser. Best for
  scripting, CI/CD, or SSH sessions.

Installation
------------

The ``[train]`` extra automatically includes all ML dependencies.

.. code-block:: bash

   /opt/zoneminder/venv/bin/pip install "pyzm[train]"

This installs all ML dependencies plus Ultralytics (YOLO), Streamlit, and the
canvas/image components used by the UI.

.. note::

   Ultralytics pulls in ``opencv-python`` from PyPI, which will shadow any
   custom OpenCV build (e.g. from source with CUDA or Apple Silicon support).
   If you need a custom OpenCV, install the training extras first, then
   replace the pip version:

   .. code-block:: bash

      pip uninstall opencv-python opencv-python-headless
      # Then build/install OpenCV from source

Launching the UI
----------------

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m pyzm.train

Or directly via Streamlit:

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m streamlit run pyzm/train/app.py -- --base-path /var/lib/zmeventnotification/models

.. important::

   Always use the **venv Python** to run the training UI. If you use the
   system Python instead, Ultralytics' auto-dependency installer will
   target ``/usr/bin/python3`` and attempt global installs with
   ``--break-system-packages``, which will fail with permission errors.

Options:

``--base-path``
   Path to your ZoneMinder models directory.
   Default: ``/var/lib/zmeventnotification/models``

``--workspace-dir``
   Override the project storage directory.
   Default: ``~/.pyzm/training``

``--processor``
   ``gpu`` or ``cpu`` for auto-detection. Default: ``gpu``

``--host``
   Bind address. Default: ``0.0.0.0``

``--port``
   Port. Default: ``8501``

Headless / CLI Training
-----------------------

For automated or server-side workflows, you can run the full training pipeline
from the command line without launching the Streamlit UI. This is useful for
scripting, CI/CD, or SSH sessions.

Basic usage:

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m pyzm.train /path/to/yolo-dataset

The dataset folder must be in standard YOLO format (``data.yaml`` + ``images/``
+ ``labels/``). The pipeline validates, imports, splits, trains, and exports
an ONNX model automatically.

Full flags:

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m pyzm.train /path/to/yolo-dataset \
       --model yolo11s \
       --epochs 100 \
       --batch 8 \
       --imgsz 640 \
       --val-ratio 0.2 \
       --device cuda:0 \
       --project-name my_project \
       --max-per-class 50 \
       --output /tmp/model.onnx

CLI options (headless mode):

``dataset`` (positional)
   Path to YOLO dataset folder.

``--model``
   Base YOLO model. Default: ``yolo11s``

``--epochs``
   Training epochs. Default: ``50``

``--batch``
   Batch size. Default: auto-detect from GPU.

``--imgsz``
   Image size. Default: ``640``

``--val-ratio``
   Train/val split ratio. Default: ``0.2``

``--output``
   ONNX export path. Default: auto in project dir.

``--project-name``
   Project name. Default: derived from dataset folder name.

``--device``
   ``auto``, ``cpu``, ``cuda:0``, etc. Default: ``auto``

``--max-per-class``
   Limit the import to at most this many images per class. Useful for
   large datasets where you want balanced representation without
   importing everything. Applied before ``--val-ratio`` splitting.
   Default: ``0`` (import all).

``--workspace-dir``
   Override the project storage directory.
   Default: ``~/.pyzm/training``

Programmatic usage:

.. code-block:: python

   from pathlib import Path
   from pyzm.train import run_pipeline

   result = run_pipeline(
       Path("/path/to/yolo-dataset"),
       epochs=50,
       model="yolo11s",
       max_per_class=50,  # 0 = import all (default)
   )
   print(f"mAP50: {result.final_mAP50:.4f}")

Workflow
--------

The training UI has three phases:

1. Select Images
^^^^^^^^^^^^^^^^

Import training images from one of three sources:

- **Pre-Annotated YOLO Dataset** -- import an existing dataset in YOLO format
  (e.g. from Roboflow). The UI reads the ``data.yaml`` and imports images with
  their annotations already attached.

- **Raw Images** -- upload unannotated images (JPG, PNG, etc.). The base model
  runs auto-detection on each image, giving you a starting point to correct.

- **ZoneMinder Events** -- connect to your ZM instance, browse monitors and
  events, and import frames directly. This is the most natural workflow for
  fixing detection problems: find an event where the model failed, import the
  frame, and correct the labels.

When the UI detects that certain classes need more training images (based on
your review corrections), it shows a banner at the top of this phase listing
which classes need attention and their current progress.

2. Review Detections
^^^^^^^^^^^^^^^^^^^^

For each imported image, review the auto-detected bounding boxes:

- **Approve** -- the detection is correct
- **Delete** -- remove a false positive
- **Rename** -- change the label (e.g. "car" should be "truck")
- **Reshape** -- drag/resize a box that's too large or misaligned
- **Add** -- draw new boxes for objects the model missed

The sidebar shows an image navigator and per-class coverage (how many images
contain each class vs. the minimum needed for training).

You can expand the canvas for more precise drawing, and clear drawn boxes if
you make a mistake.

3. Train & Export
^^^^^^^^^^^^^^^^^

Once all images are reviewed and classes have enough data:

- Configure training parameters (epochs, batch size, image size)
- The UI auto-detects GPU/CPU and suggests an appropriate batch size
- Click **Start Training** to begin fine-tuning
- A live progress bar shows epoch, loss curves, and mAP metrics
- Training logs are displayed in real time

After training completes, the UI shows:

- **Results summary** -- mAP50, mAP50-95, model size, training time
- **Per-class metrics** -- precision, recall, and AP for each class
- **Training analysis** -- interpretive guidance on the quality of your model,
  training curves, confusion matrix, F1/PR curves, and validation samples
- **Export** -- export the fine-tuned model as ONNX with a ready-to-use
  snippet for your ``objectconfig.yml``

Projects
--------

The training UI supports multiple projects. Each project stores its images,
annotations, verification state, and training runs independently under
``~/.pyzm/training/<project-name>/``.

When you launch the UI, you can create a new project or resume an existing
one. The **Switch Project** and **Reset Project** buttons in the sidebar let
you manage projects.

Testing the fine-tuned model
----------------------------

Before deploying, you can quickly verify the exported model on a test image
using the included ``test_finetuned_model.py`` example script. It runs
detection and saves an annotated image with bounding boxes drawn on it.

.. code-block:: bash

   /opt/zoneminder/venv/bin/python examples/test_finetuned_model.py \
       /path/to/test_image.jpg \
       ~/.pyzm/training/my_project/best.onnx

Class labels are automatically extracted from the ONNX metadata — no separate
labels file is needed. Optional flags:

``--labels <path>``
   Override labels with a text file (one class per line). Only needed if the
   ONNX model lacks embedded metadata.

``--confidence <float>``
   Minimum confidence threshold. Default: ``0.3``

``--out <path>``
   Output image path. Default: ``<image>_detections.<ext>``

The script prints each detection and saves the annotated image:

.. code-block:: text

   Detections: dog:94% cat:87%
     dog: 94%  [120,45 -> 380,410]
     cat: 87%  [450,200 -> 620,390]

   Saved annotated image to: test_image_detections.jpg

Using the fine-tuned model
--------------------------

After exporting, copy the ONNX file to your models directory and update your
``objectconfig.yml``:

.. code-block:: yaml

   models:
     - name: my_finetune
       type: object
       framework: opencv
       weights: /var/lib/zmeventnotification/models/custom_finetune/yolo11s_finetune.onnx
       min_confidence: 0.3
       pattern: "(dog|cat|package)"

The fine-tuned model can be used alongside your existing models. pyzm's
detection pipeline will run all configured models and merge results according
to your ``match_strategy``.

Tips
----

- **Start small** -- 10-20 images per class is enough for a first pass. You
  can always add more and retrain.
- **Use ZM events** -- importing frames from events where detection failed is
  the most effective way to improve accuracy for your specific camera angles
  and lighting.
- **Check the confusion matrix** -- it shows which classes the model confuses
  with each other, helping you decide where to add more data.
- **Watch for overfitting** -- if the best model epoch is early in training
  (e.g. epoch 15 of 50), try fewer epochs or more training data.
- **Export and test** -- use the built-in test image feature to verify the
  fine-tuned model before deploying it.
