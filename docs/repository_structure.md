# Repository Structure & Architecture Tour

This document provides a guided tour of the public repository layout, source packages, configuration system, and data documentation.

---

## 1. Top-Level Directory Layout (Public Release)

```text
multimodal-knee-oa-progression-pipeline/
├── configs/          # Declarative, versioned study and model YAML configurations
├── data/             # Public data documentation (data/README.md; no participant data)
├── docs/             # Technical methodology, pipeline engineering, and results documentation
│   ├── assets/       # Documentation-only aggregate evaluation figures (PNG)
│   ├── historical/   # Immutable transparency snapshots (historical scorer, SHA-256)
│   └── results/      # Aggregate frozen scientific JSONs, bootstrap CIs, and errata
├── release/          # Authoritative machine-readable release manifest
├── scripts/          # Release engineering and staging utilities
├── src/              # Core Python modular packages (portable research code)
├── tests/            # Automated test suite (synthetic fixtures and contract tests)
├── .github/          # Continuous integration workflows (GitHub Actions)
├── .gitignore        # Defense-in-depth protection of participant data and large binaries
├── pyproject.toml    # Project metadata, dependencies (Python >=3.13,<3.14), and build settings
└── README.md         # Executive project summary and portfolio documentation
```

*Note on Local Research Storage:* The private research environment maintains additional directories not distributed in the public release, including access-controlled raw OAI archives, intermediate processing caches, local milestone audit reports (`reports/`), and exploratory notebooks.

*Release Boundary Guarantee:* The security and privacy of the public release boundary are enforced by the combination of an explicit machine-readable allowlist (`release/public_release_manifest.json`), automated builder preflight safety checks, strict post-staging exact membership verification, and Git inspection, rather than `.gitignore` alone.

---

## 2. Source Code Architecture (`src/`)

The pipeline is organized into domain-oriented Python packages:

```text
src/
├── cohort/           # Longitudinal cohort construction and outcome candidate logic
│   ├── builder.py           # In-memory OAI cohort assembly
│   ├── feasibility.py       # Cohort attrition and eligibility tracking
│   ├── materialize.py       # Atomic materialization of cohort parquets
│   └── outcome_candidates.py # Definition of progression and replacement targets
├── data/             # Privacy-safe data inventory utilities
│   └── inventory.py         # Tabular schema and metadata scanning without row exposure
├── imaging/          # Radiograph engineering, localization, and quality control
│   ├── archive_extract.py   # Secure extraction of raw OAI DICOM archives
│   ├── dicom_inspect.py     # Metadata and pixel spacing validation
│   ├── laterality.py        # Bilateral laterality mapping and spatial verification
│   ├── preprocessing.py     # Resampling to 0.15 mm/pixel (bilinear), 160 mm physical crop
│   ├── override_crops.py    # Generation of human-adjusted override crops
│   └── final_adjudicated.py # Materialization of authoritative imaging manifest
├── modeling/         # Machine learning and deep learning pipelines
│   ├── tabular.py           # Logistic regression pipelines (Formulations A & B)
│   ├── image.py             # DenseNet121 PyTorch training and inference
│   ├── fusion.py            # Late probability fusion optimization
│   ├── final_freeze.py      # Development model and protocol freeze (Milestone 6F)
│   └── final_test_archival_recovery.py # Archival serialization recovery (Milestone 6G-B-R)
└── multimodal/       # Multimodal integration and split generation
    ├── freeze.py            # Integration of clinical, PRO, function, and imaging domains
    └── splits.py            # Deterministic participant-grouped stratified partitioning
```

*Historical Scorer Governance:* In accordance with Milestone 6H release governance, the frozen one-time final-test evaluation runner (`src/modeling/final_test_evaluation.py`) is excluded from the public executable package surface and preserved for transparency as a non-executable historical reference snapshot at [`docs/historical/final_test_evaluation.py.txt`](historical/final_test_evaluation.py.txt) (SHA-256: `fc18e711254c9b9d97dc38327bb0d101f36955bccb8fbc5930480231e3343a34`).

---

## 3. Configuration Management (`configs/`)

All scientific parameters and feature sets are defined declaratively in versioned YAML files:
- `study_v1.yaml`: Study inclusion/exclusion criteria, follow-up boundaries (V00 to V06), and primary endpoint specifications.
- `features_v1.yaml`: Clinical, demographic, WOMAC, KOOS, and physical function feature rosters.
- `split_v1.yaml`: 70/15/15 target proportions, stratification rules, and deterministic SHA-256 seed.
- `image_model_v1.yaml`: DenseNet121 architecture, ImageNet-1K weights, 320x320 resolution, learning rate schedule, and percentile intensity clipping.
- `tabular_v1.yaml`: Logistic regression hyperparameter grid, imputation strategies, and scaling rules.
- `fusion_v1.yaml`: Convex probability fusion grid ($\alpha \in [0.0, 1.0]$) and selection criteria.

---

## 4. Packaging and Distribution Boundary

The project is packaged for research transparency and public verification as **repository-checkout research software**. Installation via `pip install -e ".[dev,modeling,image_modeling]"` installs the portable execution libraries and CLI entry points. Standalone wheel distribution is not currently claimed or supported.

---

## 5. Platform Support

- **Validated Execution Environment:** macOS (Darwin arm64) using Python 3.13.2.
- **Continuous Integration:** Linux (Ubuntu x86_64) via GitHub Actions using Python 3.13.
- **Windows Notice:** Native Windows environments are not officially tested or supported; POSIX-compliant environments (macOS, Linux, WSL2) are recommended.
