# Imaging Pipeline & Quality Control: Standardized Knee Radiographs

This document describes the imaging engineering, DICOM processing, joint localization, human adjudication, and model-time transformations applied to baseline bilateral knee radiographs.

---

## 1. Imaging Pipeline Overview

The imaging pipeline converts raw bilateral digital radiographs into standardized, quality-controlled, joint-centered crops suitable for deep learning:

```mermaid
flowchart TD
    RAW_DICOM[Raw OAI Bilateral DICOM Archive] --> VALIDATE_LAT[Laterality Audit & Coordinate Validation\nScreen-Left: Anatomical Right (1)\nScreen-Right: Anatomical Left (2)]
    VALIDATE_LAT --> RESAMPLE[Pixel Spacing Normalization\nStandardize to 0.15 mm/pixel]
    RESAMPLE --> AUTO_LOC[Automatic Joint Center Localization\n160x160 mm physical square -> 1067x1067 uint16]
    AUTO_LOC --> QC_AUDIT{Adjudication Audit Engine\n978 accessions / 1,956 knee panels}
    QC_AUDIT -->|Accept: 1,534 panels| PROV_ACCEPT[Automatic Crop Approved]
    QC_AUDIT -->|Needs Override: 422 panels| MANUAL_QC[Human Reviewer Center Adjustment]
    MANUAL_QC --> MATERIALIZE[Materialize 422 Corrected Crops]
    PROV_ACCEPT & MATERIALIZE --> FINAL_MANIFEST[Authoritative Imaging Manifest: 6,961 Modeling Knees]
    FINAL_MANIFEST --> RUNTIME_TRANSFORM[Model Transform\np0.5/p99.5 clipping -> [0,1] scaling ->\n320x320 Bilinear (antialias=True) -> ImageNet Norm]
```

---

## 2. Bilateral DICOM Handling & Laterality Mapping

OAI baseline knee radiographs were acquired using fixed-flexion bilateral standing radiography (SynaFlexer frame):
- **Laterality Standard:** Following validated clinical radiograph conventions:
  - Screen-left corresponds to **anatomical RIGHT** (`knee_side_code = 1`).
  - Screen-right corresponds to **anatomical LEFT** (`knee_side_code = 2`).
- **Audit Against Unilateral Metadata:** DICOM tags in some bilateral files contain misleading unilateral anatomical indicators. The pipeline overrides these with verified anatomical spatial positions and cross-references them against baseline clinical laterality records.

---

## 3. Crop Standardization & Geometric Invariance

To standardize physical pixel spacing and spatial scale, calibrate resolution across acquisitions, and suppress background artifacts:

1. **Standardized Spatial Resolution:** Bilateral radiographs are resampled to a uniform pixel spacing of **0.150 mm per pixel** using bilinear interpolation (`Image.Resampling.BILINEAR`).
2. **Standardized Physical Field of View:** A fixed $160.05 \times 160.05\text{ mm}$ physical square window centered at the distal femoral/proximal tibial joint line is extracted:
   $$\text{Crop Dimensions} = \frac{160.05\text{ mm}}{0.150\text{ mm/pixel}} = 1,067 \times 1,067\text{ pixels}$$
3. **Bit Depth Preservation:** Raw 12-bit and 14-bit DICOM pixel intensities are preserved in uncompressed 16-bit unsigned integer format (`uint16`, range $[0, 65535]$), preventing quantization artifacts.

---

## 4. Human Quality Control & Adjudication Protocol

Automatic joint localization relies on landmark detection. To ensure localization integrity, a rigorous adjudication workflow was implemented:
- **Queued Acquisitions:** 978 acquisitions flagged by geometric, border-proximity, or contrast criteria were queued for human verification.
- **Reviewed Knee Panels:** Across the 978 queued acquisitions, **1,956 bilateral knee panels** underwent blinded human visual review.
- **Adjudication Decisions:**
  - `ACCEPT`: **1,534 panels** (automatic crop was anatomically centered and satisfactory).
  - `NEEDS_CENTER_OVERRIDE`: **422 panels** (reviewer provided corrected anatomical joint center coordinates).
  - `REJECT`: **0 panels** (no knee was deemed ungradable or corrupted).
  - `UNRESOLVED`: **0 panels** (all discrepancies were successfully resolved).
- **Override Materialization:** All 422 required override crops were generated with unchanged $160\text{ mm}$ geometry.
- **Accurate Summary:** *All queued cases were resolved with zero final rejected modeling knees; automatic localization required manual center override in 422 of 1,956 reviewed knee panels.*
- **Reviewer Context & Outcome Blinding:** Reviewers were presented with masked anonymous bilateral context, coordinate overlays, candidate joint centers, QC evidence, and joint crops to perform localization/cropping QC (rather than diagnostic grading or OA severity labeling). Reviewers were strictly **blinded to clinical progression outcomes, Kellgren-Lawrence grades, and participant metadata**.

---

## 5. Model Runtime Transformations

During dataset loading for PyTorch model training and evaluation, standardized $1,067 \times 1,067$ uint16 crops are dynamically processed through the following frozen sequence:

1. **Per-Image Percentile Clipping:**
   Robust intensity bounds are computed strictly on the individual crop:
   $$I_{\text{low}} = \text{percentile}(I, 0.5), \quad I_{\text{high}} = \text{percentile}(I, 99.5)$$
   Pixel values are clamped to $[I_{\text{low}}, I_{\text{high}}]$. This removes extreme outlier hot/dead detector pixels without distorting trabecular contrast.
2. **Min-Max Rescaling:**
   $$I_{\text{norm}} = \frac{I_{\text{clipped}} - I_{\text{low}}}{I_{\text{high}} - I_{\text{low}}} \in [0.0, 1.0]$$
3. **Spatial Resizing:**
   Resized to model input dimensions of **$320 \times 320$ pixels** using `torchvision.transforms.functional.resize`:
   $$\text{interpolation} = \text{InterpolationMode.BILINEAR}, \quad \text{antialias} = \text{True}$$
4. **Grayscale Channel Replication:**
   The single normalized grayscale channel is replicated across 3 channels ($C = 3$) to match standard deep learning vision backbones.
5. **ImageNet Normalization:**
   Channel-wise standardization using torchvision ImageNet parameters:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
6. **Training-Time Augmentation (TRAIN only):**
   Small random affine transformations: rotation $[-5^\circ, +5^\circ]$, translation $[-2\%, +2\%]$, scaling $[0.95, 1.05]$. Horizontal and vertical flips are disabled to preserve anatomical joint orientation.

---

## 6. Privacy & Artifact Integrity

- **Zero Participant Images in Documentation:** No patient radiograph crops or identifiable anatomical figures are embedded in public documentation.
- **Cryptographic Provenance:** The authoritative imaging manifest (`imaging_manifest.parquet`) references 6,961 knees with individual file hashes, ensuring complete cryptographic traceability.
