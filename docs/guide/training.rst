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

Or directly via Streamlit (note: ``--host`` / ``--port`` are not available
in this form — use ``python -m pyzm.train`` instead):

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m streamlit run pyzm/train/app.py -- --base-path /var/lib/zmeventnotification/models

.. important::

   Always use the **venv Python** to run the training UI. If you use the
   system Python instead, Ultralytics' auto-dependency installer will
   target ``/usr/bin/python3`` and attempt global installs with
   ``--break-system-packages``, which will fail with permission errors.

Options (for ``python -m pyzm.train`` without a dataset argument):

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
       --mode new_class \
       --output /tmp/model.onnx

CLI options (shared by headless and ``--correct`` modes):

``dataset`` (positional)
   Path to YOLO dataset folder (headless mode only).

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

``--mode``
   Fine-tuning mode. ``new_class`` for teaching the model an object it has
   never seen (minimal augmentation — mosaic off for small datasets).
   ``refine`` for improving detection of a class the model already knows
   (moderate augmentation). Default: ``new_class``

``--workspace-dir``
   Override the project storage directory.
   Default: ``~/.pyzm/training``

``--base-path``
   Path to your ZoneMinder models directory (used to locate the base
   model for ``--correct`` mode).
   Default: ``/var/lib/zmeventnotification/models``

``--processor``
   ``gpu`` or ``cpu`` for auto-detection. Default: ``gpu``

``--min-confidence``
   Minimum confidence threshold for auto-detection (``--correct`` mode).
   Default: ``0.3``

Correct Model (headless)
^^^^^^^^^^^^^^^^^^^^^^^^

For the common case where you have images the model gets wrong, the
``--correct`` flag runs the full correct-and-retrain workflow in one command:

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m pyzm.train --correct /path/to/images

This scans the folder for images, runs the current model on each one,
auto-approves all detections, and retrains with ``mode=refine``.  No
``data.yaml`` is needed — just a folder of images.

You can combine ``--correct`` with most of the standard flags:

.. code-block:: bash

   /opt/zoneminder/venv/bin/python -m pyzm.train --correct /path/to/images \
       --model yolo11s \
       --epochs 100 \
       --batch 8 \
       --base-path /var/lib/zmeventnotification/models \
       --processor gpu

Programmatic usage:

.. code-block:: python

   from pathlib import Path
   from pyzm.train import run_pipeline

   result = run_pipeline(
       Path("/path/to/yolo-dataset"),
       epochs=50,
       model="yolo11s",
       max_per_class=50,  # 0 = import all (default)
       mode="new_class",  # or "refine" for existing classes
   )
   print(f"mAP50: {result.final_mAP50:.4f}")

For the correct-model workflow:

.. code-block:: python

   from pathlib import Path
   from pyzm.train import run_correct_pipeline

   result = run_correct_pipeline(
       Path("/path/to/images"),
       model="yolo11s",
       epochs=100,
   )
   print(f"mAP50: {result.final_mAP50:.4f}")

Workflow
--------

The training UI has three phases:

1. Select Images
^^^^^^^^^^^^^^^^

Import training images using one of two modes:

- **Manually annotate** -- upload images or point at a server folder. Toggle
  **Auto-detect objects at import** to have the model pre-detect objects,
  giving you a starting point to correct. You can also run detection later
  from the Review phase using the "Detect all" button. Either way, you
  review every image: approve correct detections, delete false positives,
  rename mislabeled objects, reshape inaccurate boxes, and draw boxes for
  objects the model missed.

- **Import pre-annotated dataset** -- import a pre-annotated dataset in YOLO
  format (e.g. from Roboflow). The UI reads the ``data.yaml`` and imports
  images with their annotations already attached. You can optionally filter
  out classes you don't need using the **Classes to ignore** selector —
  annotations for ignored classes are skipped and class IDs are remapped
  automatically.

This phase is organized into three collapsible sub-steps — **Base model**,
**Images**, and **Review images** — each showing a green checkmark when
complete. When you reopen an existing project, the base model and dataset
path are automatically restored and completed steps remain collapsed.

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

The sidebar shows an image navigator, per-class coverage (how many images
contain each class vs. the minimum needed for training), and a **Bulk Approve**
slider that approves all pending detections above a confidence threshold in one
click.

Additional review tools:

- **Filter bar** -- filter the image grid by status (all / approved /
  unapproved) and by object class.
- **Re-detect** -- re-run the model on a single image to refresh detections.
- **Re-review** -- reopen an already-reviewed image for further edits.
- **Remove Image** -- remove an image from the project entirely.
- **Expand canvas** -- enlarge the drawing area for more precise box placement.
  Clear drawn boxes if you make a mistake.

3. Train & Export
^^^^^^^^^^^^^^^^^

Once all images are reviewed and classes have enough data:

- Configure training parameters (epochs, batch size, image size)
- Select the **fine-tuning mode**:

  - **New class** — for teaching the model an object it has never seen
    (e.g. gun, knife). Uses minimal augmentation so the model learns clean
    representations first. Mosaic is off for small datasets.
  - **Refine existing** — for improving detection of a class the model
    already knows (e.g. adding your camera's person images). Uses moderate
    augmentation to help generalise.

- The UI auto-detects GPU/CPU and suggests an appropriate batch size
- **Adaptive hyperparameters** are displayed — freeze layers, learning rate,
  mosaic, erasing, and patience are all tuned automatically based on
  dataset size and fine-tuning mode. The backbone is always frozen and
  cosine LR is always enabled to prevent pretrained feature destruction.
- Click **Start Training** to begin fine-tuning
- A live progress bar shows epoch, loss curves, and mAP metrics
- Training logs are displayed in real time

After training completes:

- **ONNX export runs automatically** — no manual export step needed. The
  exported path is displayed in the results.
- **Results summary** -- mAP50, mAP50-95, model size, training time
- **Per-class metrics** -- precision, recall, and AP for each class
- **Training analysis** -- interpretive guidance on the quality of your model,
  training curves, confusion matrix, F1/PR curves, and validation samples
- **Test image** -- upload an image to verify the fine-tuned model's
  detections before deploying

Annotation Strategy
-------------------

Choosing which detections to keep during the Review phase depends on your
fine-tuning mode and how you plan to deploy the model.

What to Label (New Class Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When teaching the model a completely new object (e.g. package, gun, knife):

- Label **only** your new class.
- **Remove** detections for classes the base model already knows (person, car,
  dog, etc.). The base model was trained on millions of images and will always
  outperform a model fine-tuned on 20–500 images for those classes.
- The Review UI automatically identifies which classes the base model already
  knows and shows a contextual recommendation with a one-click **"Delete all
  known labels"** button to remove them in bulk.

What to Label (Refine Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When correcting the model's mistakes on classes it already knows:

- Label **all** classes — you're improving the model's existing knowledge.
- Approve correct detections, fix wrong ones (rename, reshape, delete false
  positives).

Background Images
^^^^^^^^^^^^^^^^^

Images where all detections are deleted become **negative/background** examples.
These teach the model "nothing is here" — useful for reducing false positives.
A few background images are helpful but shouldn't dominate the dataset.

Production Deployment
^^^^^^^^^^^^^^^^^^^^^

The fine-tuned model only detects the classes you trained it on. To get full
coverage, run both models in your ``objectconfig.yml``:

.. code-block:: yaml

   models:
     - name: base
       type: object
       framework: opencv
       weights: /path/to/yolo11s.onnx

     - name: custom
       type: object
       framework: opencv
       weights: /path/to/my_finetune.onnx

   same_model_sequence_strategy: union

The base model handles standard objects; the fine-tuned model adds your custom
class. The ``union`` strategy merges detections from both models.

How Many Images?
^^^^^^^^^^^^^^^^

- **10–50 images**: Good starting point. Label only new classes. Remove
  known-class detections.
- **50–200 images**: Solid results for new classes. Still remove known-class
  detections.
- **200–500 images**: Strong model. Known-class detections won't help much.
- **500+ images**: Large dataset. Could theoretically keep known classes, but
  the union strategy is simpler and more effective.

Projects
--------

The training UI supports multiple projects. Each project stores its images,
annotations, verification state, and training runs independently under
``~/.pyzm/training/<project-name>/``.

When you launch the UI, you can create a new project or resume an existing
one. The **Switch Project** and **Delete Project** buttons in the sidebar let
you manage projects.

Adaptive Fine-Tuning
--------------------

Training hyperparameters are automatically tuned based on your dataset size
and fine-tuning mode. This ensures that fine-tuning never overwrites
pretrained feature representations.

**Dataset size tiers:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - Tier
     - Images
     - Freeze
     - Learning Rate
     - Patience
   * - Small
     - < 200
     - 10 layers
     - 0.0005
     - 10
   * - Medium
     - 200–999
     - 10 layers
     - 0.001
     - 15
   * - Large
     - 1,000–4,999
     - 5 layers
     - 0.002
     - 20
   * - XLarge
     - 5,000+
     - 3 layers
     - 0.005
     - 25

All tiers freeze backbone layers and use cosine LR annealing.

**Fine-tuning modes:**

- ``new_class`` — minimal augmentation. Mosaic is off for small/medium
  datasets (< 1,000 images) because stitching 4 images of an unknown object
  creates unrealistic composites. Erasing is off or very low.
- ``refine`` — moderate augmentation from the start, since the model already
  understands the object structure. Mosaic 0.3–0.8, erasing 0.1–0.3.

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
- **Import with auto-detect** -- enable "Auto-detect objects at import" and
  point at images where your model is wrong. Fix the mistakes, then retrain.
  This is the fastest way to improve accuracy for your specific cameras and
  lighting conditions.
- **Check the confusion matrix** -- it shows which classes the model confuses
  with each other, helping you decide where to add more data.
- **Watch for overfitting** -- if the best model epoch is early in training
  (e.g. epoch 15 of 50), try fewer epochs or more training data.
- **Export and test** -- use the built-in test image feature to verify the
  fine-tuned model before deploying it.
