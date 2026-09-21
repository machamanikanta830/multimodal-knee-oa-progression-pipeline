# Milestone 6D: image-only development

The three predeclared image candidates completed on NVIDIA A100-SXM4-40GB CUDA.
Canonical production freeze: `data/processed/modeling/image/v1/production_cuda/run_freeze.json`.
Candidate 2 is the selected **image development** candidate for later fusion. This is not a
final project model, not a locked TEST evaluation, and not authorization to start fusion
training. Local portable-preparation artifacts remain non-scientific readiness records.

## Inputs and protection

`src/modeling/image.py` verifies the approved dataset, participant split, knee
split, feature-group and imaging-manifest SHA-256 values before using them.
Identifiers are `participant_id`, `knee_side_code` (`1` = right, `2` = left),
and `baseline_visit` (`V00`). The imaging manifest uses anatomical labels `R`/`L`.
The frozen target is `composite_progression`. Image references are
`effective_crop_relative_path`, `effective_crop_sha256`, and
`effective_provenance`, relative to `data/processed/oai_images/v3_frozen_full`.
None of the latter provenance information is used as a disease predictor.

Only TRAIN (4,873 knees, 749 events) and VALIDATION (1,044 knees, 161 events)
enter image datasets. TEST metadata are checked structurally by the existing
frozen-split validator; TEST images cannot enter either model dataset and TEST
predictions/metrics are prohibited. Every development image is checked for SHA,
uint16 dtype and 1067-square shape before each production candidate. Each model
image load additionally checks membership and image SHA before opening the array.

The preservation baseline at
`data/processed/modeling/image/v1/pre_change_preservation.json` covers the
original scientific snapshot, frozen multimodal/split bundles, and both current
and retained superseded tabular artifact bundles. Preservation hashing is a
separate scientific integrity audit, not a model image loader or evaluation.

## Preprocessing and augmentation

The existing frozen `imaging.preprocessing.robust_minmax` implements image-local
float32 0.5th/99.5th percentile clipping and min-max scaling. No population
statistics or fitted preprocessing are used. Degenerate arrays transform safely
to zero, but are rejected by the production integrity gate. There is no CLAHE.
Resize directly to 320 square using bilinear antialiasing; repeat the grayscale
channel three times. Apply mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]` from
`DenseNet121_Weights.IMAGENET1K_V1`. The weights' 224-square spatial recipe is
explicitly overridden by the scientifically specified 320-square input.

TRAIN only: one bilinear affine, rotation within +/-5 degrees, translation at
most 2% per axis, scale 0.95–1.05, zero fill. No horizontal/vertical flip, random
crop, image augmentation outside TRAIN, or outcome-derived transform. Affine RNG
is keyed deterministically to seed, epoch and internal knee identity. VALIDATION
has only deterministic preprocessing. A montage is exported without participant
identifiers or private imaging URLs.

## Predeclared training

`configs/image_model_v1.yaml` is declared before production fitting. Batch size
2, workers 0, unweighted BCEWithLogitsLoss, AdamW, weight decay 1e-4:

| Candidate | Backbone | Learning rate | Maximum epochs |
| --- | --- | --- | --- |
| 0 | Frozen, including BatchNorm running statistics | 1e-3 | 10 |
| 1 | Full fine-tuning | 1e-4 | 25 |
| 2 | Full fine-tuning | 3e-5 | 25 |

There is no head warm-up. Validation AUROC drives stopping, patience 5 after the
predeclared minimum of 3 epochs. Best epoch/candidate ties use higher average
precision, lower Brier, lower log loss, then candidate order. The checkpoint is
reloaded and its validation metrics verified before accepting results.

Model, classifier, NumPy, Python, augmentation and DataLoader seed: **63026**.
Participant-clustered bootstrap seed: **63027**, 1,000 replicates. All knees from
each sampled validation participant are included. Percentile 95% intervals cover
AUROC, average precision and Brier. Calibration intercept/slope and plots are
descriptive; no final recalibration model is fitted.

Strict PyTorch deterministic algorithms are enabled, with deterministic cuDNN
and a CUBLAS workspace setting. CUDA uses float16 AMP; MPS/CPU use float32.
Unsupported deterministic operations fail instead of silently weakening the
policy. Cross-device or library-version bitwise equality is not promised. CUDA
execution requires verification on the target GPU; local synthetic CPU tests
do not establish a CUDA reproducibility guarantee.

## Compute readiness and execution

An eight-image balanced TRAIN-only frozen-feature/head overfit check must learn
above chance with decreasing loss. A short full-backbone gradient benchmark uses
the exact architecture, 320 resolution and batch size. Before benchmarking, the
local maximum projected compute budget is fixed at two hours. Projection treats
all 60 possible epochs and TRAIN/VALIDATION passes as full-training equivalents;
it is conservative and excludes decoding/I/O, not an exact duration forecast.
Timing observations are reused, not required to repeat bitwise. Their provenance
and montage hash must still match.

Prepare locally without production fitting:

```sh
python -m modeling.image --device mps --output data/processed/modeling/image/v1/portable_preparation_v1
```

For a private Colab T4 environment, mount the retained repository/data tree with
its original relative paths, including the protected-artifact baseline and
tabular bundles; run from the repository root. Do not publish NDA data or place
private S3 URLs in a public notebook. Install the pinned extras in a fresh
compatible Python environment and confirm CUDA is available:

```sh
python -m pip install -e '.[modeling,image_modeling,dev]'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
python -m modeling.image --device cuda --train --output data/processed/modeling/image/v1/production_cuda
```

Use the distinct production directory: runtime-specific preparation provenance
is immutable and must not be overwritten with a different device's metadata.
The same production command rerun validates the frozen artifact catalog and
reproduces selected VALIDATION predictions without retraining. A partial
production run without a freeze is retained and refused for blind overwrite;
an explicitly new run directory is required. A failed integrity/provenance check
is a blocker, not permission to create a different split or repair source data.

Production stores each history/best checkpoint, candidate results, selected
candidate, complete unique validation predictions, metrics, bootstrap samples
and intervals, calibration plot/data, input/config/source hashes and freeze.
No model fitting, calibration, threshold selection or metric uses TEST.
Image-only candidate selection is not a final project-model declaration and
does not authorize fusion or locked TEST evaluation.

## A100 production freeze and controlled import

Production fitting ran in the private CUDA execution bundle on **NVIDIA A100-SXM4-40GB**,
PyTorch `2.14.0+cu130`, torchvision `0.29.0+cu130`, CUDA runtime `13.0`. The independently
verified return archive SHA-256 is
`1cac894dd869840ff344d11e71fcf5a8ee6984753bb7c681f86e7f84fc336645`.

Approved `modeling.colab_bundle validate-return` passed against the staged directory
without training or image inference (`validation_rows` 1,044; `TEST_scored` false).
Canonical destination `data/processed/modeling/image/v1/production_cuda/` did not exist.
Import used the existing `modeling.colab_bundle.copy_exact` guard: 28 files, zero
staging/import SHA-256 mismatches. Selected checkpoint `candidate_2/best.pt` SHA-256 is
`b14a74f329a268ae6855a9dcf69f6850d38e31f7fb9bd9fc0f3336394a2dec1c`.

| Candidate | Backbone | LR | Best epoch | Epochs run | AUROC | AUPRC | Brier | log loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Frozen backbone, including BatchNorm running statistics | 1e-3 | 8 | 10 | 0.6373951028045273 | 0.24623423901200328 | 0.1290867687262441 | 0.42990549191231964 |
| 1 | Full fine-tuning | 1e-4 | 1 | 7 | 0.598327272215696 | 0.21336337467359964 | 0.1306694909111938 | 0.43202691458765624 |
| 2 selected | Full fine-tuning | 3e-5 | 5 | 10 | 0.6778557008504321 | 0.29510337110727886 | 0.1254394713614726 | 0.41772200542828164 |

Selected candidate 2 VALIDATION diagnostics: 1,044 observations, 161 events, prevalence
0.15421455938697318; calibration intercept -0.28002146584095894, slope 0.6574188820805513
(descriptive; not applied). Participant-clustered percentile 95% CIs from the existing
1,000-replicate bootstrap (seed 63027): AUROC 0.6253939841771126–0.7248299706266542;
AUPRC 0.2326535556750379–0.3737863561705084; Brier 0.1088099046467949–0.143337277120727.
All 1,000 replicates were valid. TEST remained absent from the execution bundle
(`TEST_images` 0, loader opens 0, `TEST_scored` false); no TEST predictions, metrics,
calibration, threshold selection or TEST-derived model selection exist.

Full numeric closeout, provenance specifications, and candidate diagnostics are documented in
`docs/modeling_and_evaluation.md` and `docs/final_results.md`.
