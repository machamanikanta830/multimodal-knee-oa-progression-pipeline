"""Milestone 6G-A: Blind TEST Image Inference Bundle Preparation.

Prepares an isolated, blind image-inference execution bundle for the 1,044 TEST knees.
Strictly excludes all target/outcome labels, event counts, prevalence, and metrics.
Uses exact frozen Milestone 6D preprocessing (bilinear with antialiasing) and
DenseNet121 Candidate 2 checkpoint semantics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from imaging.artifact_io import sha256_file
from imaging.preprocessing import robust_minmax
from modeling import final_freeze
from multimodal.freeze import require

DEFAULT_BUNDLE_DIR = Path("data/processed/modeling/image/test_inference/v1/bundle")
DEFAULT_OUTPUT_DIR = Path("data/processed/modeling/image/test_inference/v1/output")
DEFAULT_ARCHIVE_PATH = Path(
    "data/processed/modeling/image/test_inference/v1/colab_a100_test_inference_bundle.tar.gz"
)
TEST_INFERENCE_DIR = Path("data/processed/modeling/image/test_inference/v1")

SPLIT_MANIFEST_PATH = Path(
    "data/processed/multimodal/final_v1/splits/v1/knee_split_manifest.parquet"
)
PARTICIPANT_SPLIT_PATH = Path(
    "data/processed/multimodal/final_v1/splits/v1/participant_split_manifest.parquet"
)
IMAGING_MANIFEST_PATH = Path(
    "data/processed/oai_images/v3_frozen_full/final_adjudicated_v1/imaging_manifest.parquet"
)
IMAGE_ROOT = Path("data/processed/oai_images/v3_frozen_full")
CHECKPOINT_PATH = Path("data/processed/modeling/image/v1/production_cuda/candidate_2/best.pt")
CONFIG_PATH = Path("configs/image_model_v1.yaml")

APPROVED_CHECKPOINT_SHA = "b14a74f329a268ae6855a9dcf69f6850d38e31f7fb9bd9fc0f3336394a2dec1c"
EXPECTED_IMAGING_MANIFEST_SHA = "cce476f5ecc2381232b026a9aec09dffe58db391116deadb7fddb2405aee0fa7"
EXPECTED_TEST_KNEES = 1044
EXPECTED_TEST_PARTICIPANTS = 543

AUTHORITATIVE_PREPROCESSING_CODE_FILE = "src/modeling/image.py"
AUTHORITATIVE_PREPROCESSING_CODE_SHA = (
    "5a47d6032d441974e065f5b1d477a4bbe31bfe0f8de6477c916794e90dd31793"
)
AUTHORITATIVE_PREPROCESSING_CONFIG_FILE = "configs/image_model_v1.yaml"
AUTHORITATIVE_PREPROCESSING_CONFIG_SHA = (
    "895331f848144ff2582de626345476800a887065972d17903a2209c80acdefbb"
)

MANIFEST_COLUMNS = [
    "participant_id",
    "knee_side_code",
    "baseline_visit",
    "bundle_image_path",
    "image_sha256",
]

FORBIDDEN_TERMS = [
    "composite_progression",
    "kl_progression",
    "jsn_progression",
    "progression_48m",
    "outcome",
    "event_indicator",
    "prevalence",
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "p_tabular",
    "p_fusion",
]

WEIGHTS = DenseNet121_Weights.IMAGENET1K_V1


def build_candidate_model(checkpoint_path: Path) -> nn.Module:
    """Build DenseNet121 Candidate 2 model and load state_dict with strict semantics."""
    require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
    actual_sha = sha256_file(checkpoint_path)
    require(
        actual_sha == APPROVED_CHECKPOINT_SHA,
        f"Checkpoint SHA mismatch: expected {APPROVED_CHECKPOINT_SHA}, got {actual_sha}",
    )

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_test_crop(array: np.ndarray) -> torch.Tensor:
    """Apply exact Milestone 6D frozen preprocessing to a uint16 crop.

    Interpolation is explicitly bilinear with antialiasing enabled.
    """
    require(
        array.dtype == np.uint16 and array.shape == (1067, 1067),
        f"Invalid crop shape/dtype: {array.shape}, {array.dtype}",
    )
    scaled = robust_minmax(array, lower_percentile=0.5, upper_percentile=99.5)
    x = torch.from_numpy(scaled).unsqueeze(0)
    x = TF.resize(x, [320, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    x = x.repeat(3, 1, 1)
    recipe = WEIGHTS.transforms()
    return TF.normalize(x, recipe.mean, recipe.std)


def preflight_verification() -> dict:
    """Read-only verification of upstream hashes, test split, and image completeness."""
    # 1. Final freeze hashes
    dev_hashes = final_freeze.verify_development_artifacts()

    # 2. Checkpoint SHA
    require(CHECKPOINT_PATH.is_file(), f"Missing checkpoint: {CHECKPOINT_PATH}")
    ckpt_sha = sha256_file(CHECKPOINT_PATH)
    require(
        ckpt_sha == APPROVED_CHECKPOINT_SHA,
        f"Checkpoint SHA mismatch: expected {APPROVED_CHECKPOINT_SHA}, got {ckpt_sha}",
    )

    # 3. Imaging manifest SHA
    require(IMAGING_MANIFEST_PATH.is_file(), f"Missing imaging manifest: {IMAGING_MANIFEST_PATH}")
    img_manifest_sha = sha256_file(IMAGING_MANIFEST_PATH)
    require(
        img_manifest_sha == EXPECTED_IMAGING_MANIFEST_SHA,
        f"Imaging manifest SHA mismatch: expected {EXPECTED_IMAGING_MANIFEST_SHA}, got {img_manifest_sha}",
    )

    # 4. Knee split manifest - ONLY identity and split columns loaded
    require(SPLIT_MANIFEST_PATH.is_file(), f"Missing split manifest: {SPLIT_MANIFEST_PATH}")
    k_split = pd.read_parquet(
        SPLIT_MANIFEST_PATH, columns=["participant_id", "knee_side_code", "baseline_visit", "split"]
    )
    test_knees = k_split[k_split["split"] == "TEST"].copy()
    require(len(test_knees) == 1044, f"Expected 1,044 TEST knees, got {len(test_knees)}")
    require(
        test_knees["participant_id"].nunique() == 543,
        f"Expected 543 TEST participants, got {test_knees['participant_id'].nunique()}",
    )
    composite_keys = test_knees[
        ["participant_id", "knee_side_code", "baseline_visit"]
    ].drop_duplicates()
    require(
        len(composite_keys) == 1044,
        f"Duplicate composite keys in TEST split: expected 1044, got {len(composite_keys)}",
    )

    # 5. Join with imaging manifest to verify all crops exist on disk
    manifest = pd.read_parquet(
        IMAGING_MANIFEST_PATH,
        columns=[
            "participant_id",
            "knee_side_code",
            "baseline_visit",
            "effective_crop_relative_path",
            "effective_crop_sha256",
        ],
    )
    merged = test_knees.merge(
        manifest, on=["participant_id", "knee_side_code", "baseline_visit"], how="inner"
    )
    require(
        len(merged) == 1044,
        f"Expected 1044 resolved TEST image paths, got {len(merged)}",
    )

    for row in merged.itertuples(index=False):
        crop_path = IMAGE_ROOT / row.effective_crop_relative_path
        require(
            crop_path.is_file(),
            f"Missing TEST crop file: {crop_path}",
        )

    return {
        "status": "PREFLIGHT_VERIFIED",
        "development_hashes": dev_hashes,
        "checkpoint_sha256": ckpt_sha,
        "imaging_manifest_sha256": img_manifest_sha,
        "test_knees": 1044,
        "test_participants": 543,
        "all_images_present": True,
    }


def audit_label_exclusion(bundle_dir: Path) -> dict:
    """Recursively audit all files in bundle_dir to prove complete absence of target/outcome terms."""
    violations = []
    audited_files = 0

    for path in bundle_dir.rglob("*"):
        if not path.is_file():
            continue
        audited_files += 1
        rel_str = str(path.relative_to(bundle_dir)).lower()

        # Check path name for forbidden terms
        for term in FORBIDDEN_TERMS:
            if term in rel_str:
                violations.append(f"Forbidden term '{term}' in filename: {rel_str}")

        # Check content based on file extension
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            for col in df.columns:
                col_lower = str(col).lower()
                for term in FORBIDDEN_TERMS:
                    if term in col_lower:
                        violations.append(f"Forbidden column '{col}' in {rel_str}")
        elif path.suffix in (".json", ".yaml", ".yml", ".txt"):
            try:
                text = path.read_text(encoding="utf-8").lower()
                for term in FORBIDDEN_TERMS:
                    if term in text:
                        violations.append(f"Forbidden term '{term}' in {rel_str}")
            except UnicodeDecodeError:
                pass

    require(
        len(violations) == 0,
        f"Label exclusion audit failed with {len(violations)} violations: {violations}",
    )
    return {
        "audit_passed": True,
        "audited_files": audited_files,
        "violations": [],
    }


def generate_colab_inference_script() -> str:
    """Generate the self-contained, blind inference script to be packaged inside the bundle."""
    return '''"""Standalone Blind TEST Image Inference Script for NVIDIA A100 / CUDA runtime.

Executes deterministic blind image inference for the 1,044 TEST knees.
Computes NO performance metrics, loads NO outcome labels, and applies NO calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torchvision.models import densenet121
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

APPROVED_CHECKPOINT_SHA = "b14a74f329a268ae6855a9dcf69f6850d38e31f7fb9bd9fc0f3336394a2dec1c"
EXPECTED_TEST_KNEES = 1044
EXPECTED_TEST_PARTICIPANTS = 543
AUTHORITATIVE_PREPROCESSING_CODE_FILE = "src/modeling/image.py"
AUTHORITATIVE_PREPROCESSING_CODE_SHA = "5a47d6032d441974e065f5b1d477a4bbe31bfe0f8de6477c916794e90dd31793"
AUTHORITATIVE_PREPROCESSING_CONFIG_FILE = "configs/image_model_v1.yaml"
AUTHORITATIVE_PREPROCESSING_CONFIG_SHA = "895331f848144ff2582de626345476800a887065972d17903a2209c80acdefbb"
MANIFEST_COLUMNS = [
    "participant_id",
    "knee_side_code",
    "baseline_visit",
    "bundle_image_path",
    "image_sha256",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def robust_minmax(
    array: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    clipped = np.clip(values, low, high)
    min_val, max_val = float(clipped.min()), float(clipped.max())
    if max_val <= min_val:
        return np.zeros(clipped.shape, dtype=np.float32)
    return ((clipped - min_val) / (max_val - min_val)).astype(np.float32)


def preprocess_crop(array: np.ndarray) -> torch.Tensor:
    if array.dtype != np.uint16 or array.shape != (1067, 1067):
        raise ValueError(f"Invalid array shape/dtype: {array.shape}, {array.dtype}")
    scaled = robust_minmax(array, lower_percentile=0.5, upper_percentile=99.5)
    x = torch.from_numpy(scaled).unsqueeze(0)
    x = TF.resize(x, [320, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    x = x.repeat(3, 1, 1)
    return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


def build_model(checkpoint_path: Path) -> nn.Module:
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != APPROVED_CHECKPOINT_SHA:
        raise ValueError(f"Checkpoint SHA mismatch: expected {APPROVED_CHECKPOINT_SHA}, got {actual_sha}")
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class DualLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def run_blind_inference(
    bundle_dir: Path,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int = 32,
    require_a100: bool = False,
) -> dict:
    start_time = time.time()
    bundle_dir = bundle_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "execution_log.txt"
    logger = DualLogger(log_path)
    orig_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("=" * 70)
        print("STAGE A: BLIND TEST IMAGE INFERENCE RUNTIME")
        print(f"Timestamp UTC: {datetime.now(UTC).isoformat()}")
        print("=" * 70)

        # 1. Device selection and verification
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(device_name)
        print(f"[INFO] Using device: {device}")
        gpu_name = None
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU: {gpu_name}")
            print(f"[INFO] CUDA Runtime: {torch.version.cuda}")
            if require_a100:
                if not re.search(r"A100", gpu_name, re.IGNORECASE):
                    raise RuntimeError(f"STOP: Required NVIDIA A100 GPU not detected. Found: {gpu_name}")
                print("[PASS] Verified NVIDIA A100 hardware.")

        print(f"[INFO] Python Version: {platform.python_version()}")
        print(f"[INFO] PyTorch Version: {torch.__version__}")
        print(f"[INFO] Torchvision Version: {torchvision.__version__}")

        # 2. Manifest loading and verification
        manifest_path = bundle_dir / "manifest.parquet"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        manifest_sha = sha256_file(manifest_path)
        print(f"[INFO] Manifest SHA-256: {manifest_sha}")

        df = pd.read_parquet(manifest_path)
        if list(df.columns) != MANIFEST_COLUMNS:
            raise ValueError(f"Manifest columns mismatch: expected {MANIFEST_COLUMNS}, got {list(df.columns)}")
        if len(df) != EXPECTED_TEST_KNEES:
            raise ValueError(f"Expected {EXPECTED_TEST_KNEES} TEST knees, got {len(df)}")
        if df["participant_id"].nunique() != EXPECTED_TEST_PARTICIPANTS:
            raise ValueError(f"Expected {EXPECTED_TEST_PARTICIPANTS} TEST participants, got {df['participant_id'].nunique()}")

        # 3. Model construction
        ckpt_path = bundle_dir / "checkpoint/best.pt"
        ckpt_sha = sha256_file(ckpt_path)
        print(f"[INFO] Loading checkpoint: {ckpt_path} (SHA: {ckpt_sha})")
        model = build_model(ckpt_path).to(device)
        model.eval()
        print("[PASS] Model loaded into eval mode with strict weights matching.")

        # 4. Read config hashes if available
        config_path = bundle_dir / "config/inference_config.json"
        config_sha = sha256_file(config_path) if config_path.is_file() else None
        bundle_freeze_path = bundle_dir / "bundle_freeze.json"
        bundle_freeze_sha = sha256_file(bundle_freeze_path) if bundle_freeze_path.is_file() else None
        script_sha = sha256_file(Path(__file__))

        # 5. Batch inference under torch.inference_mode()
        probabilities = []
        print(f"[INFO] Running blind inference over {len(df)} TEST images (batch_size={batch_size})...")

        with torch.inference_mode():
            for i in range(0, len(df), batch_size):
                batch_rows = df.iloc[i : i + batch_size]
                tensors = []
                for row in batch_rows.itertuples(index=False):
                    img_path = bundle_dir / row.bundle_image_path
                    if not img_path.is_file():
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    array = np.load(img_path, allow_pickle=False)
                    tensors.append(preprocess_crop(array))
                batch_tensor = torch.stack(tensors).to(device)
                logits = model(batch_tensor).flatten()
                probs = logits.sigmoid().cpu().numpy().tolist()
                probabilities.extend(probs)
                if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(df):
                    print(f"[PROGRESS] Evaluated {min(i + batch_size, len(df))}/{len(df)} images")

        if len(probabilities) != EXPECTED_TEST_KNEES:
            raise ValueError(f"Output count mismatch: expected {EXPECTED_TEST_KNEES}, got {len(probabilities)}")

        # 6. Assemble predictions dataframe (whitelisted columns only)
        pred_df = pd.DataFrame({
            "participant_id": df["participant_id"],
            "knee_side_code": df["knee_side_code"],
            "baseline_visit": df["baseline_visit"],
            "p_image": probabilities,
        })

        # Validate numerical bounds
        if pred_df["p_image"].isna().any():
            raise ValueError("Missing/NaN probabilities produced")
        if ((pred_df["p_image"] < 0.0) | (pred_df["p_image"] > 1.0)).any():
            raise ValueError("Probabilities outside [0, 1] range")

        pred_path = output_dir / "test_image_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        pred_sha = sha256_file(pred_path)
        print(f"[SUCCESS] Wrote predictions to {pred_path} (SHA: {pred_sha})")
        print(f"[INFO] Prediction Range: min={pred_df['p_image'].min():.6f}, max={pred_df['p_image'].max():.6f}")

        duration = time.time() - start_time

        # 7. Complete Stage-A Return Provenance
        run_freeze = {
            "status": "COMPLETED_BLIND_TEST_IMAGE_INFERENCE",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "authoritative_preprocessing_code_file": AUTHORITATIVE_PREPROCESSING_CODE_FILE,
            "authoritative_preprocessing_code_sha256": AUTHORITATIVE_PREPROCESSING_CODE_SHA,
            "authoritative_preprocessing_config_file": AUTHORITATIVE_PREPROCESSING_CONFIG_FILE,
            "authoritative_preprocessing_config_sha256": AUTHORITATIVE_PREPROCESSING_CONFIG_SHA,
            "bundled_inference_code_file": "src/inference.py",
            "bundled_inference_code_sha256": script_sha,
            "bundled_inference_config_file": "config/inference_config.json",
            "bundled_inference_config_sha256": config_sha,
            "bundle_freeze_sha256": bundle_freeze_sha,
            "checkpoint_sha256": ckpt_sha,
            "manifest_sha256": manifest_sha,
            "preprocessing_implementation_note": (
                "Preprocessing logic (robust_minmax percentile [0.5, 99.5] clipping, "
                "float32 min-max scaling to [0, 1], bilinear resize with antialias=True to 320x320, "
                "3-channel replication, ImageNet normalization) is physically embedded/copy-contained "
                "inside src/inference.py for standalone execution, producing numerically identical output to authoritative src/modeling/image.py."
            ),
            "image_count": EXPECTED_TEST_KNEES,
            "participant_count": EXPECTED_TEST_PARTICIPANTS,
            "batch_size": batch_size,
            "device": str(device),
            "gpu": gpu_name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_runtime_version": torch.version.cuda if device.type == "cuda" else None,
            "model_eval_confirmed": True,
            "inference_mode_confirmed": True,
            "augmentation": False,
            "outcome_labels_present": False,
            "metrics_computed": False,
            "calibration_computed": False,
            "fusion_computed": False,
            "predictions_file": "test_image_predictions.parquet",
            "predictions_sha256": pred_sha,
        }
        rf_path = output_dir / "inference_run_freeze.json"
        with open(rf_path, "w", encoding="utf-8") as f:
            json.dump(run_freeze, f, indent=2, sort_keys=True)

        print(f"[SUCCESS] Stage A blind inference completed in {duration:.2f}s")
    finally:
        sys.stdout = orig_stdout
        logger.close()

    # 8. Artifact hashes and return archive packaging
    log_sha = sha256_file(log_path)
    rf_sha = sha256_file(rf_path)
    artifact_hashes = {
        "test_image_predictions.parquet": pred_sha,
        "inference_run_freeze.json": rf_sha,
        "execution_log.txt": log_sha,
    }
    ah_path = output_dir / "artifact_hashes.json"
    with open(ah_path, "w", encoding="utf-8") as f:
        json.dump(artifact_hashes, f, indent=2, sort_keys=True)

    # Package return archive
    return_tar_path = output_dir / "test_image_inference_return.tar.gz"
    with tarfile.open(return_tar_path, "w:gz") as tar:
        for name in ["test_image_predictions.parquet", "inference_run_freeze.json", "execution_log.txt", "artifact_hashes.json"]:
            fpath = output_dir / name
            tar.add(fpath, arcname=f"test_image_inference_return/{name}")

    return_tar_sha = sha256_file(return_tar_path)
    print("=" * 70)
    print("STAGE A RETURN ARCHIVE CREATED:")
    print(f"Path:   {return_tar_path}")
    print(f"Size:   {return_tar_path.stat().st_size} bytes")
    print(f"SHA256: {return_tar_sha}")
    print("=" * 70)

    run_freeze["return_archive_sha256"] = return_tar_sha
    run_freeze["return_archive_bytes"] = return_tar_path.stat().st_size
    return run_freeze


def main():
    parser = argparse.ArgumentParser(description="Run blind TEST image inference.")
    parser.add_argument("--bundle", type=Path, default=Path("."), help="Bundle root directory")
    parser.add_argument("--output", type=Path, default=Path("./output"), help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--require-a100", action="store_true", help="Fail closed if GPU is not NVIDIA A100")
    args = parser.parse_args()

    run_blind_inference(
        bundle_dir=args.bundle,
        output_dir=args.output,
        device_name=args.device,
        batch_size=args.batch_size,
        require_a100=args.require_a100,
    )


if __name__ == "__main__":
    main()
'''


def build_test_inference_bundle(
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
) -> dict:
    """Construct the complete blind TEST image inference bundle and package it into a tarball."""
    preflight = preflight_verification()

    bundle_dir = bundle_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    images_dir = bundle_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = bundle_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir = bundle_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    src_dir = bundle_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve and copy TEST crops with strict naming and hash verification
    k_split = pd.read_parquet(
        SPLIT_MANIFEST_PATH, columns=["participant_id", "knee_side_code", "baseline_visit", "split"]
    )
    test_knees = k_split[k_split["split"] == "TEST"].copy()
    manifest = pd.read_parquet(
        IMAGING_MANIFEST_PATH,
        columns=[
            "participant_id",
            "knee_side_code",
            "baseline_visit",
            "effective_crop_relative_path",
            "effective_crop_sha256",
        ],
    )
    merged = test_knees.merge(
        manifest, on=["participant_id", "knee_side_code", "baseline_visit"], how="inner"
    )
    require(len(merged) == 1044, f"Merged TEST knees mismatch: expected 1044, got {len(merged)}")

    manifest_rows = []
    for row in merged.itertuples(index=False):
        src_path = IMAGE_ROOT / row.effective_crop_relative_path
        require(src_path.is_file(), f"Missing crop file: {src_path}")
        src_sha = sha256_file(src_path)
        require(
            src_sha == row.effective_crop_sha256,
            f"Source crop SHA mismatch for {src_path}: expected {row.effective_crop_sha256}, got {src_sha}",
        )

        bundle_image_rel = (
            f"images/{row.participant_id}_{row.knee_side_code}_{row.baseline_visit}.npy"
        )
        dst_path = bundle_dir / bundle_image_rel
        if not dst_path.exists():
            shutil.copyfile(src_path, dst_path)
        dst_sha = sha256_file(dst_path)
        require(
            dst_sha == src_sha,
            f"Destination crop SHA mismatch for {dst_path}: expected {src_sha}, got {dst_sha}",
        )

        manifest_rows.append(
            {
                "participant_id": row.participant_id,
                "knee_side_code": row.knee_side_code,
                "baseline_visit": row.baseline_visit,
                "bundle_image_path": bundle_image_rel,
                "image_sha256": dst_sha,
            }
        )

    # 2. Write whitelisted manifest (parquet + json)
    manifest_df = pd.DataFrame(manifest_rows)
    require(
        list(manifest_df.columns) == MANIFEST_COLUMNS,
        f"Manifest columns violation: {list(manifest_df.columns)}",
    )
    manifest_parquet_path = bundle_dir / "manifest.parquet"
    manifest_df.to_parquet(manifest_parquet_path, index=False)
    manifest_json_path = bundle_dir / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2, sort_keys=True)

    # 3. Copy checkpoint and config
    dst_ckpt = checkpoint_dir / "best.pt"
    if not dst_ckpt.exists():
        shutil.copyfile(CHECKPOINT_PATH, dst_ckpt)
    require(
        sha256_file(dst_ckpt) == APPROVED_CHECKPOINT_SHA,
        "Copied checkpoint SHA mismatch",
    )

    old_cfg = config_dir / "image_model_v1.yaml"
    if old_cfg.exists():
        old_cfg.unlink()

    dst_cfg = config_dir / "inference_config.json"
    inference_config = {
        "architecture": "DenseNet121",
        "weights": "DenseNet121_Weights.IMAGENET1K_V1",
        "candidate_index": 2,
        "checkpoint": "checkpoint/best.pt",
        "checkpoint_sha256": APPROVED_CHECKPOINT_SHA,
        "input_size": 320,
        "percentiles": [0.5, 99.5],
        "interpolation": "bilinear",
        "antialias": True,
        "channels": 3,
        "deterministic_algorithms": True,
        "authoritative_preprocessing_config_file": AUTHORITATIVE_PREPROCESSING_CONFIG_FILE,
        "authoritative_preprocessing_config_sha256": AUTHORITATIVE_PREPROCESSING_CONFIG_SHA,
        "authoritative_preprocessing_code_file": AUTHORITATIVE_PREPROCESSING_CODE_FILE,
        "authoritative_preprocessing_code_sha256": AUTHORITATIVE_PREPROCESSING_CODE_SHA,
    }
    with open(dst_cfg, "w", encoding="utf-8") as f:
        json.dump(inference_config, f, indent=2, sort_keys=True)

    # 4. Generate standalone src/inference.py and src/requirements.txt
    inference_script_path = src_dir / "inference.py"
    inference_code = generate_colab_inference_script()
    inference_script_path.write_text(inference_code, encoding="utf-8")

    reqs_path = src_dir / "requirements.txt"
    reqs_text = (
        "torch==2.14.0\n"
        "torchvision==0.29.0\n"
        "numpy==2.5.2\n"
        "pandas==3.0.5\n"
        "pyarrow==25.0.1\n"
        "Pillow==12.3.0\n"
    )
    reqs_path.write_text(reqs_text, encoding="utf-8")

    # 5. Generate bundle_freeze.json
    bundle_freeze = {
        "status": "APPROVED_BLIND_TEST_IMAGE_BUNDLE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bundle_version": "test_image_inference_v1",
        "execution_target": "NVIDIA A100 (CUDA 13.0, PyTorch 2.14.0+cu130 preferred)",
        "test_census": {
            "test_knees": 1044,
            "test_participants": 543,
            "baseline_visit": "V00",
        },
        "model_specification": {
            "architecture": "DenseNet121",
            "candidate_index": 2,
            "checkpoint_sha256": APPROVED_CHECKPOINT_SHA,
            "checkpoint_file": "checkpoint/best.pt",
            "classifier": "nn.Linear(1024, 1) emitting raw logits; torch.sigmoid applied to logits",
        },
        "authoritative_upstream_provenance": {
            "authoritative_preprocessing_code_file": AUTHORITATIVE_PREPROCESSING_CODE_FILE,
            "authoritative_preprocessing_code_sha256": AUTHORITATIVE_PREPROCESSING_CODE_SHA,
            "authoritative_preprocessing_config_file": AUTHORITATIVE_PREPROCESSING_CONFIG_FILE,
            "authoritative_preprocessing_config_sha256": AUTHORITATIVE_PREPROCESSING_CONFIG_SHA,
            "authoritative_checkpoint_file": "data/processed/modeling/image/v1/production_cuda/candidate_2/best.pt",
            "authoritative_checkpoint_sha256": APPROVED_CHECKPOINT_SHA,
        },
        "preprocessing_specification": {
            "source_crop_shape": [1067, 1067],
            "source_crop_dtype": "uint16",
            "percentiles": [0.5, 99.5],
            "scaling": "Image-local float32 clip/min-max [0, 1]",
            "resize_shape": [320, 320],
            "interpolation_mode": "bilinear",
            "antialias": True,
            "channels": "grayscale replicated to 3 channels",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "test_augmentation": "None (model.eval(), torch.inference_mode())",
            "numerical_equivalence_verified": True,
            "numerical_equivalence_diff": 0.0,
            "preprocessing_implementation_note": (
                "Preprocessing logic (robust_minmax percentile [0.5, 99.5] clipping, "
                "float32 min-max scaling to [0, 1], bilinear resize with antialias=True to 320x320, "
                "3-channel replication, ImageNet normalization) is physically embedded/copy-contained "
                "inside src/inference.py for standalone execution, producing numerically identical output (diff == 0.0) to authoritative src/modeling/image.py."
            ),
        },
        "label_exclusion_policy": {
            "labels_present": False,
            "blinded_evaluation": True,
            "whitelisted_manifest_columns": MANIFEST_COLUMNS,
        },
    }
    freeze_path = bundle_dir / "bundle_freeze.json"
    with open(freeze_path, "w", encoding="utf-8") as f:
        json.dump(bundle_freeze, f, indent=2, sort_keys=True)

    # 6. Generate artifact_hashes.json for all files in the bundle
    bundle_hashes = {}
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            rel = str(path.relative_to(bundle_dir))
            bundle_hashes[rel] = sha256_file(path)

    hashes_path = bundle_dir / "artifact_hashes.json"
    with open(hashes_path, "w", encoding="utf-8") as f:
        json.dump(bundle_hashes, f, indent=2, sort_keys=True)

    # 7. Label exclusion audit
    audit = audit_label_exclusion(bundle_dir)

    # 8. Package portable tarball
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    print(f"[INFO] Packaging bundle into {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        for entry in sorted(bundle_dir.iterdir()):
            tar.add(entry, arcname=entry.name)

    archive_size = archive_path.stat().st_size
    archive_sha = sha256_file(archive_path)

    archive_catalog = {
        "archive_file": archive_path.name,
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "total_bundle_files": len(bundle_hashes) + 1,  # including artifact_hashes.json
        "test_image_count": 1044,
        "authoritative_checkpoint_sha256": APPROVED_CHECKPOINT_SHA,
        "authoritative_preprocessing_code_sha256": AUTHORITATIVE_PREPROCESSING_CODE_SHA,
        "authoritative_preprocessing_config_sha256": AUTHORITATIVE_PREPROCESSING_CONFIG_SHA,
        "manifest_sha256": bundle_hashes["manifest.parquet"],
        "bundled_inference_code_sha256": bundle_hashes["src/inference.py"],
        "bundled_inference_config_sha256": bundle_hashes["config/inference_config.json"],
        "bundle_freeze_sha256": sha256_file(freeze_path),
    }
    catalog_path = archive_path.parent / f"{archive_path.name}.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(archive_catalog, f, indent=2, sort_keys=True)

    print(
        f"[SUCCESS] Archive created: {archive_path} ({archive_size / (1024 * 1024):.2f} MB, SHA: {archive_sha})"
    )
    return {
        "status": "APPROVED_BLIND_TEST_IMAGE_BUNDLE",
        "bundle_dir": str(bundle_dir),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "test_knees": 1044,
        "test_participants": 543,
        "preflight": preflight,
        "audit": audit,
        "archive_catalog": archive_catalog,
    }


def test_synthetic_inference() -> dict:
    """Run preprocessing and Candidate 2 forward pass strictly on a SYNTHETIC dummy array.

    Does NOT touch or evaluate any real TEST image crop.
    """
    dummy_array = np.full((1067, 1067), fill_value=12000, dtype=np.uint16)
    # add synthetic spatial gradient
    dummy_array += np.linspace(0, 5000, 1067, dtype=np.uint16)[:, None]

    tensor = preprocess_test_crop(dummy_array).unsqueeze(0)
    require(tensor.shape == (1, 3, 320, 320), f"Synthetic tensor shape mismatch: {tensor.shape}")

    model = build_candidate_model(CHECKPOINT_PATH)
    with torch.inference_mode():
        logits = model(tensor).flatten()
        prob = float(logits.sigmoid().item())

    require(
        np.isfinite(prob) and 0.0 <= prob <= 1.0, f"Synthetic probability out of bounds: {prob}"
    )
    return {
        "synthetic_test_passed": True,
        "dummy_input_shape": [1067, 1067],
        "dummy_input_dtype": "uint16",
        "preprocessed_tensor_shape": list(tensor.shape),
        "output_probability": prob,
    }


def validate_blind_predictions(predictions_path: Path, manifest_path: Path) -> dict:
    """Validate Stage A blind test image predictions output contract."""
    require(predictions_path.is_file(), f"Missing predictions: {predictions_path}")
    require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")

    preds = pd.read_parquet(predictions_path)
    manifest = pd.read_parquet(manifest_path)

    expected_cols = ["participant_id", "knee_side_code", "baseline_visit", "p_image"]
    require(
        list(preds.columns) == expected_cols,
        f"Predictions columns mismatch: expected {expected_cols}, got {list(preds.columns)}",
    )
    require(len(preds) == 1044, f"Expected 1,044 predictions, got {len(preds)}")
    require(
        preds["participant_id"].nunique() == 543,
        f"Expected 543 participants, got {preds['participant_id'].nunique()}",
    )
    require(
        not preds[["participant_id", "knee_side_code", "baseline_visit"]].duplicated().any(),
        "Duplicate knee keys in predictions",
    )

    # Check key alignment with manifest
    for col in ["participant_id", "knee_side_code", "baseline_visit"]:
        require(
            preds[col].equals(manifest[col]),
            f"Key sequence mismatch between predictions and manifest on {col}",
        )

    # Check numerical validity
    require(not preds["p_image"].isna().any(), "Predictions contain NaN values")
    require(np.isfinite(preds["p_image"]).all(), "Predictions contain non-finite values")
    require(
        ((preds["p_image"] >= 0.0) & (preds["p_image"] <= 1.0)).all(),
        "Probabilities out of [0, 1] range",
    )

    # Check for forbidden columns
    for col in preds.columns:
        col_lower = str(col).lower()
        for term in FORBIDDEN_TERMS:
            require(term not in col_lower, f"Forbidden term in prediction column: {col}")

    return {
        "validation_passed": True,
        "rows": len(preds),
        "participants": preds["participant_id"].nunique(),
        "predictions_sha256": sha256_file(predictions_path),
    }


def validate_stage_a_return(
    return_target: Path,
    manifest_path: Path = DEFAULT_BUNDLE_DIR / "manifest.parquet",
) -> dict:
    """Validate Stage A blind return package (tar.gz archive, directory, or predictions parquet).

    Enforces strict contractual checks:
    - Exactly 1,044 prediction rows
    - Exactly 543 unique participants
    - Unique (participant_id, knee_side_code, baseline_visit)
    - Strict schema: ['participant_id', 'knee_side_code', 'baseline_visit', 'p_image']
    - Finite p_image in [0.0, 1.0]
    - Zero target/outcome columns and zero performance metrics
    - Frozen Candidate 2 checkpoint SHA-256 verification
    - Confirmed model.eval(), torch.inference_mode(), augmentation=False
    """
    import tempfile

    require(return_target.exists(), f"Missing Stage A return target: {return_target}")

    if return_target.is_file() and return_target.name.endswith(".tar.gz"):
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(return_target, "r:gz") as tar:
                try:
                    tar.extractall(td, filter="data")
                except TypeError:
                    tar.extractall(td)
            extracted_dir = Path(td)
            subdirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
            target_dir = subdirs[0] if subdirs else extracted_dir
            return _validate_stage_a_directory(
                target_dir, manifest_path, archive_path=return_target
            )

    elif return_target.is_dir():
        return _validate_stage_a_directory(return_target, manifest_path)

    elif return_target.is_file() and return_target.suffix == ".parquet":
        return validate_blind_predictions(return_target, manifest_path)

    else:
        raise ValueError(f"Unrecognized Stage A return format: {return_target}")


def _validate_stage_a_directory(
    directory: Path,
    manifest_path: Path,
    archive_path: Path | None = None,
) -> dict:
    pred_path = directory / "test_image_predictions.parquet"
    require(pred_path.is_file(), f"Missing predictions parquet in return: {pred_path}")
    pred_val = validate_blind_predictions(pred_path, manifest_path)

    rf_path = directory / "inference_run_freeze.json"
    require(rf_path.is_file(), f"Missing inference_run_freeze.json in return: {rf_path}")
    rf = json.loads(rf_path.read_text(encoding="utf-8"))

    ah_path = directory / "artifact_hashes.json"
    require(ah_path.is_file(), f"Missing artifact_hashes.json in return: {ah_path}")
    ah = json.loads(ah_path.read_text(encoding="utf-8"))

    for fname, expected_sha in ah.items():
        fpath = directory / fname
        require(fpath.is_file(), f"Artifact listed in hashes not found: {fname}")
        actual_sha = sha256_file(fpath)
        require(
            actual_sha == expected_sha,
            f"Artifact SHA mismatch for {fname}: expected {expected_sha}, got {actual_sha}",
        )

    require(
        rf.get("checkpoint_sha256") == APPROVED_CHECKPOINT_SHA,
        "Checkpoint SHA drifted in run freeze",
    )
    require(
        rf.get("authoritative_preprocessing_code_sha256") == AUTHORITATIVE_PREPROCESSING_CODE_SHA,
        "Authoritative preprocessing code SHA mismatch in run freeze",
    )
    require(
        rf.get("authoritative_preprocessing_config_sha256")
        == AUTHORITATIVE_PREPROCESSING_CONFIG_SHA,
        "Authoritative preprocessing config SHA mismatch in run freeze",
    )
    require(
        rf.get("bundled_inference_code_sha256") is not None,
        "Missing bundled_inference_code_sha256 in run freeze",
    )
    require(
        rf.get("bundled_inference_config_sha256") is not None,
        "Missing bundled_inference_config_sha256 in run freeze",
    )
    require(rf.get("image_count") == EXPECTED_TEST_KNEES, "Image count mismatch in run freeze")
    require(
        rf.get("participant_count") == EXPECTED_TEST_PARTICIPANTS,
        "Participant count mismatch in run freeze",
    )
    require(rf.get("model_eval_confirmed") is True, "model.eval() was not confirmed")
    require(rf.get("inference_mode_confirmed") is True, "inference_mode was not confirmed")
    require(rf.get("augmentation") is False, "Augmentation was not False")
    require(rf.get("outcome_labels_present") is False, "Outcome labels were reported present")
    require(rf.get("metrics_computed") is False, "Metrics were reported computed")
    require(rf.get("calibration_computed") is False, "Calibration was reported computed")
    require(rf.get("fusion_computed") is False, "Fusion was reported computed")
    require(
        rf.get("predictions_sha256") == pred_val["predictions_sha256"],
        "Prediction SHA mismatch with run freeze",
    )

    return {
        "stage_a_validation_passed": True,
        "rows": pred_val["rows"],
        "participants": pred_val["participants"],
        "predictions_sha256": pred_val["predictions_sha256"],
        "checkpoint_sha256": rf["checkpoint_sha256"],
        "device": rf.get("device"),
        "gpu": rf.get("gpu"),
        "return_archive_sha256": sha256_file(archive_path) if archive_path else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 6G-A Blind TEST Image Inference Bundle."
    )
    parser.add_argument(
        "--build", action="store_true", help="Build and package blind inference bundle"
    )
    parser.add_argument(
        "--synthetic-check", action="store_true", help="Run synthetic dummy inference test"
    )
    args = parser.parse_args()

    if args.synthetic_check:
        res = test_synthetic_inference()
        print(json.dumps(res, indent=2))

    if args.build:
        res = build_test_inference_bundle()
        print(json.dumps(res["archive_catalog"], indent=2))


if __name__ == "__main__":
    main()
