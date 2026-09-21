# Milestone 6D: private Colab CUDA execution bundle

This derived execution bundle contains **only 4,873 TRAIN and 1,044 VALIDATION
effective crops**. TEST images, TEST identifiers/labels, full-cohort tables, raw
DICOM archives, prior crops and unrelated project data are absent. It is not a
new cohort or split. Keep the archive private: it contains participant IDs,
outcome labels and research images. Never publish it or use public Drive sharing.

All crops are exact byte copies, not transformed crops. The manifest and the two
split projections are explicitly derived artifacts. Original full-source hashes
were recomputed and checked on the Mac when packaging. They are retained as
provenance expectations, not falsely claimed as hashes of the smaller projections.
On Colab, the trusted transfer archive SHA plus the bundle freeze/catalog
authenticate those derived inputs; every included artifact/image is checked.

The complete approved `modeling.image` and imaging preprocessing source files
are unchanged. A separate adapter changes only the filesystem/data-loading and
preservation boundaries. Metric, calibration, bootstrap and membership helpers
are extracted verbatim from the approved source, with parent/block SHA lineage.
All architecture, configuration, transforms, loss, search, stopping, selection,
seeds and production output generation remain those of the approved implementation.

## Colab setup

Select **Runtime -> Change runtime type -> A100 GPU** when available; **L4** and
**T4** are also approved. The CUDA gate prefers A100, then L4, then T4 among
visible approved devices, and production provenance records the detected name.
Upload the archive to private Drive at `MyDrive/knee_oa_6d/colab_cuda_bundle.tar.gz`.
The extracted directory retains its original `colab_t4_bundle` name for filesystem
compatibility only. Mount Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Verify the archive against the authoritative SHA printed in the Mac final report
**before extracting or executing bundled scripts**:

```bash
%%bash
set -euo pipefail
expected_archive_sha='PASTE_THE_AUTHORITATIVE_ARCHIVE_SHA_FROM_THE_FINAL_REPORT'
printf '%s  %s\n' "$expected_archive_sha" /content/drive/MyDrive/knee_oa_6d/colab_cuda_bundle.tar.gz | sha256sum --check -
test ! -e /content/colab_t4_bundle
tar -xzf /content/drive/MyDrive/knee_oa_6d/colab_cuda_bundle.tar.gz -C /content
bash /content/colab_t4_bundle/scripts/colab_setup.sh
```

The setup script reports GPU/driver, installs **exact pinned versions** in a
separate environment, checks versions/dependency compatibility/approved CUDA GPU, validates
all bundle hashes and 5,917 image hashes, verifies roles/counts/linkage, then trains.
Any incompatible version, driver, missing CUDA, non-approved GPU or integrity failure stops.
There is no older-library, CPU, smaller-resolution, different-model or TEST
fallback. The official CUDA wheel build suffix is recorded; its base torch and
torchvision versions must remain exactly 2.14.0 and 0.29.0. The pinned PyTorch
Linux distribution declares CUDA 13 dependencies; an incompatible Colab driver
must cause STOP, not a switch to another PyTorch version. Target installation and
production training remain to be exercised on the actual approved CUDA GPU.

Manual verification/training after setup dependencies are installed:

```bash
cd /content/colab_t4_bundle
/content/knee_oa_6d_env/bin/python scripts/verify_bundle.py --cuda
/content/knee_oa_6d_env/bin/python scripts/run_training.py --device cuda --train --output data/processed/modeling/image/v1/production_cuda
```

`scripts/run_training.py --plan` verifies command/path resolution without fitting
or inference. Actual execution requires A100/L4/T4 CUDA. The adapter delegates to the
unchanged `modeling.image.main` and production functions. Before every fit the
bundle and each development image are revalidated. No model loader has any TEST
image reference. Do not add TEST files to this directory.

## Scientific settings (unchanged)

uint16 1067-square input; per-image 0.5/99.5 clipping/min-max; 320-square bilinear
antialias resize; repeat grayscale; exact ImageNet normalization. TRAIN affine
only: +/-5 degrees, <=2% translation, scale .95–1.05, no flips. DenseNet121
IMAGENET1K_V1, binary logit, unweighted BCE, AdamW, weight decay 1e-4, batch 2,
workers 0. Candidate 0 frozen backbone LR1e-3/max10; candidate 1 full LR1e-4/max25;
candidate 2 full LR3e-5/max25. No head warm-up. Minimum 3 epochs, patience 5,
VALIDATION AUROC stopping/selection; ties AP/Brier/log-loss/order. Model/loader/
augmentation seed 63026; participant bootstrap seed 63027, 1,000 replicates.
No fusion, resplitting, outcome changes, or locked test evaluation.

## Outputs and return

The output directory is
`data/processed/modeling/image/v1/production_cuda` inside this bundle. It contains
candidate histories/best checkpoints/input audits, selected candidate, 1,044
unique VALIDATION predictions, metrics, calibration plot/data, bootstrap samples/
intervals, source/config/weight/adapter provenance and `run_freeze.json`. No TEST
predictions are created. Re-running an identical frozen production validates and
reproduces VALIDATION predictions, without retraining. An unmarked partial run
is retained and refused for blind overwrite; contact the project owner rather
than deleting it or expanding the candidate search.

Return only the production directory, not the input crops or virtual environment:

```bash
%%bash
set -euo pipefail
cd /content/colab_t4_bundle/data/processed/modeling/image/v1
tar -czf /content/drive/MyDrive/knee_oa_6d/production_cuda.tar.gz production_cuda
sha256sum /content/drive/MyDrive/knee_oa_6d/production_cuda.tar.gz
```

Record the return archive SHA independently, download, and verify it before
extracting to a new staging directory. From the original Mac repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m modeling.colab_bundle validate-return --bundle data/processed/modeling/image/v1/colab_t4_bundle --output /absolute/path/to/staged/production_cuda
```

This read-only validator checks the local original sources/preservation,
bundle linkage and hashes, every returned artifact/checkpoint provenance,
configuration/candidate/epoch consistency, all validation prediction IDs/labels,
metrics, selected objective and bootstrap/calibration completeness. It does not
train, perform image inference or create TEST predictions. A successful
validation is a prerequisite to a separately authorized import; this command
does not overwrite local artifacts. Keep the original and returned freeze
records rather than rewriting provenance to Mac runtime values.
