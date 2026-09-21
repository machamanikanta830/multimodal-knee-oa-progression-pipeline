"""Frozen-source DenseNet121 development: TRAIN fitting, VALIDATION scoring, never TEST."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from imaging.artifact_io import file_lock, sha256_file
from imaging.preprocessing import robust_minmax
from modeling import tabular
from multimodal.freeze import load_config, publish_json, publish_parquet, read_json, require

OUTPUT = Path("data/processed/modeling/image/v1")
CONFIG = Path("configs/image_model_v1.yaml")
IMAGE_ROOT = Path("data/processed/oai_images/v3_frozen_full")
MANIFEST = IMAGE_ROOT / "final_adjudicated_v1/imaging_manifest.parquet"
IMAGE_SHA = "cce476f5ecc2381232b026a9aec09dffe58db391116deadb7fddb2405aee0fa7"
WEIGHTS = DenseNet121_Weights.IMAGENET1K_V1
IMAGE_COLUMNS = ("effective_crop_relative_path", "effective_crop_sha256", "effective_provenance")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(c: dict) -> None:
    require(
        c["architecture"] == "DenseNet121"
        and c["weights"] == "DenseNet121_Weights.IMAGENET1K_V1"
        and c["target"] == tabular.TARGET
        and c["input_size"] == 320
        and c["percentiles"] == [0.5, 99.5],
        "Scientific model/input policy changed",
    )
    require(c["loss"] == "unweighted_BCEWithLogitsLoss", "Class-weight policy changed")
    require(c["optimizer"] == "AdamW" and c["weight_decay"] == 1e-4, "Optimizer changed")
    require(
        c["candidates"]
        == [
            {"index": 0, "frozen_backbone": True, "lr": 1e-3, "epochs": 10},
            {"index": 1, "frozen_backbone": False, "lr": 1e-4, "epochs": 25},
            {"index": 2, "frozen_backbone": False, "lr": 3e-5, "epochs": 25},
        ],
        "Predeclared candidate space changed",
    )
    require(
        c["augmentation"]
        == {
            "degrees": 5.0,
            "translation": 0.02,
            "scale": [0.95, 1.05],
            "horizontal_flip": False,
            "vertical_flip": False,
            "fill": 0.0,
        },
        "Unsafe augmentation policy",
    )
    require(
        c["selection"]
        == ["AUROC_desc", "AUPRC_desc", "Brier_asc", "log_loss_asc", "candidate_index_asc"]
        and c["patience"] == 5
        and c["minimum_epochs"] == 3
        and c["head_warmup_epochs"] == 0
        and c["test_prediction_permitted"] is False
        and c["deterministic_algorithms"] is True,
        "Selection/determinism/TEST policy changed",
    )


def specifications(c: dict) -> tuple[dict, dict]:
    transform = WEIGHTS.transforms()
    return {
        "source": "Frozen effective uint16 crop; never raw DICOM or uint8 preview",
        "dtype": "uint16",
        "source_shape": [1067, 1067],
        "percentiles": c["percentiles"],
        "scaling": "Image-local float32 clip/min-max [0,1]",
        "degenerate_image": "Transform returns zeros safely; production integrity gate rejects",
        "resize": [320, 320],
        "interpolation": "bilinear",
        "antialias": True,
        "channels": "repeat grayscale three times",
        "mean": transform.mean,
        "std": transform.std,
        "weight_identifier": c["weights"],
        "weight_spatial_recipe_override": "320 square, no 224 center crop",
        "dataset_fitted_statistics": False,
        "CLAHE": False,
    }, {
        **c["augmentation"],
        "partition": "TRAIN only",
        "interpolation": "bilinear",
        "rng": "Per seed/epoch/internal knee identity",
        "operation": "single conservative affine",
    }


def resolve_image(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    require(
        isinstance(relative, str) and not Path(relative).is_absolute(), "Invalid image reference"
    )
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Image path escapes frozen root")
    require(path.is_file(), "Effective image missing")
    return path


@dataclass
class ImageData:
    development: tabular.DevelopmentData
    root: Path

    def validate(self) -> None:
        self.development.validate()
        for part in (self.development.train, self.development.validation):
            require(set(IMAGE_COLUMNS) <= set(part.frame), "Missing effective image metadata")
            require(part.frame[list(IMAGE_COLUMNS)].notna().all().all(), "Unresolved image")
            require(
                part.frame.effective_provenance.isin(
                    ["AUTO_PASS", "HUMAN_ACCEPT", "HUMAN_OVERRIDE"]
                ).all(),
                "Unresolved provenance",
            )
        refs = pd.concat([self.development.train.frame, self.development.validation.frame])
        require(
            not refs.effective_crop_relative_path.duplicated().any(), "Duplicate effective image"
        )


def load_image_data() -> ImageData:
    require(sha256_file(MANIFEST) == IMAGE_SHA, "Approved imaging manifest SHA mismatch")
    development = tabular.load_development()
    # Only metadata are read for TEST; no TEST image path enters a Dataset.
    source = pd.read_parquet(
        tabular.DEFAULT_DIRECTORY / "final_multimodal_dataset.parquet",
        columns=[*tabular.IDENTITY, *IMAGE_COLUMNS],
    )
    manifest = pd.read_parquet(MANIFEST)
    require(not manifest.duplicated(list(tabular.IDENTITY)).any(), "Duplicate image linkage")
    for part in (development.train, development.validation):
        linked = part.frame[list(tabular.IDENTITY)].merge(
            source, on=list(tabular.IDENTITY), how="left", validate="one_to_one", sort=False
        )
        check = linked.merge(
            manifest[[*tabular.IDENTITY, *IMAGE_COLUMNS, "anatomical_side"]],
            on=list(tabular.IDENTITY),
            validate="one_to_one",
            suffixes=("", "_manifest"),
            how="left",
            sort=False,
        )
        for c in IMAGE_COLUMNS:
            require(check[c].eq(check[c + "_manifest"]).all(), "Cohort/imaging linkage mismatch")
            part.frame[c] = linked[c].to_numpy()
        require(
            check.anatomical_side.eq(check.knee_side_code.map({"1": "R", "2": "L"})).all(),
            "Anatomical side mismatch",
        )
    development.source_hashes["imaging_manifest"] = IMAGE_SHA
    data = ImageData(development, IMAGE_ROOT)
    data.validate()
    return data


def check_array(array: np.ndarray) -> None:
    require(
        array.dtype == np.uint16 and array.shape == (1067, 1067), "Frozen crop dtype/shape mismatch"
    )


def integrity(data: ImageData) -> dict:
    data.validate()
    result = {}
    for part in (data.development.train, data.development.validation):
        minimum, maximum, degenerate = 65535, 0, 0
        records = []
        for row in part.frame.itertuples(index=False):
            path = resolve_image(data.root, row.effective_crop_relative_path)
            actual = sha256_file(path)
            require(actual == row.effective_crop_sha256, "Effective image SHA mismatch")
            array = np.load(path, allow_pickle=False)
            check_array(array)
            minimum, maximum = min(minimum, int(array.min())), max(maximum, int(array.max()))
            low, high = np.percentile(array.astype(np.float32), [0.5, 99.5])
            degenerate += int(high <= low)
            records.append([row.participant_id, row.knee_side_code, row.baseline_visit, actual])
        require(degenerate == 0, "Degenerate production effective image")
        result[part.name] = {
            "images": len(records),
            "events": int(part.labels().sum()),
            "dtype": "uint16",
            "shape": [1067, 1067],
            "minimum": minimum,
            "maximum": maximum,
            "degenerate": degenerate,
            "identity_SHA_catalog": digest({"images": records}),
            "missing_or_SHA_mismatches": 0,
        }
    return {"partitions": result, "TEST_images_loaded_by_model_loader": 0, "participant_overlap": 0}


def transform_image(
    array: np.ndarray, config: dict, *, train: bool = False, key: tuple = (), epoch: int = 0
) -> torch.Tensor:
    check_array(array)
    x = torch.from_numpy(robust_minmax(array)).unsqueeze(0)
    x = TF.resize(x, [320, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if train:
        aug = config["augmentation"]
        seed = int.from_bytes(
            hashlib.sha256(repr((config["seed"], epoch, key)).encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        angle = float(rng.uniform(-aug["degrees"], aug["degrees"]))
        shift = [
            int(round(v))
            for v in rng.uniform(-aug["translation"] * 320, aug["translation"] * 320, size=2)
        ]
        scale = float(rng.uniform(*aug["scale"]))
        x = TF.affine(
            x, angle, shift, scale, [0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0.0
        )
    x = x.repeat(3, 1, 1)
    recipe = WEIGHTS.transforms()
    return TF.normalize(x, recipe.mean, recipe.std)


class KneeImages(Dataset):
    def __init__(
        self, partition: tabular.DevelopmentPartition, root: Path, config: dict, *, augment: bool
    ):
        partition.validate()
        require(not augment or partition.name == "TRAIN", "Validation augmentation forbidden")
        self.partition, self.root, self.config, self.augment, self.epoch = (
            partition,
            root,
            config,
            augment,
            0,
        )

    def __len__(self):
        return len(self.partition.frame)

    def __getitem__(self, index):
        # Membership is rechecked before any image open, even after mutation of the frame.
        row = self.partition.frame.iloc[index]
        require(
            self.partition.name in ("TRAIN", "VALIDATION")
            and self.partition.membership[row.participant_id] == self.partition.name,
            "TEST/outside-partition image forbidden",
        )
        key = tuple(row[c] for c in tabular.IDENTITY)
        require(key in self.partition.expected_keys, "Unexpected image knee")
        path = resolve_image(self.root, row.effective_crop_relative_path)
        require(sha256_file(path) == row.effective_crop_sha256, "Effective image SHA mismatch")
        image = np.load(path, allow_pickle=False)
        x = transform_image(image, self.config, train=self.augment, key=key, epoch=self.epoch)
        return x, torch.tensor(float(row[tabular.TARGET]), dtype=torch.float32), index


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset: KneeImages, config: dict, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["workers"],
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )


def device_for(name: str = "auto") -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    require(name in ("cpu", "cuda", "mps"), "Unsupported device")
    require(name != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    require(name != "mps" or torch.backends.mps.is_available(), "MPS unavailable")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_info(device: torch.device) -> dict:
    return {
        "device": str(device),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "CUDA_runtime": torch.version.cuda,
        "CUDA_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "MPS_available": torch.backends.mps.is_available(),
        "determinism": "Strict deterministic algorithms; unsupported operations fail, never silently downgraded. Cross-device/version bit equality is not guaranteed.",
    }


def pretrained_path(output: Path) -> Path:
    path = output / "pretrained" / Path(WEIGHTS.url).name
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = path.stem.rsplit("-", 1)[1]
    if not path.exists():
        torch.hub.download_url_to_file(WEIGHTS.url, str(path), hash_prefix=prefix, progress=False)
    require(sha256_file(path).startswith(prefix), "Pretrained weights SHA mismatch")
    catalog = {
        "identifier": "DenseNet121_Weights.IMAGENET1K_V1",
        "file": str(path.relative_to(output)),
        "sha256": sha256_file(path),
        "upstream_hash_prefix": prefix,
    }
    publish_json(catalog, output / "pretrained_weights.json")
    return path


def build_model(candidate: dict, weights: Path | None) -> nn.Module:
    model = densenet121(weights=None)
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # torchvision's historical LuaTorch checkpoint requires its documented key migration.
        import re

        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        state = {pattern.sub(r"\1\2", k): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    for parameter in model.features.parameters():
        parameter.requires_grad_(not candidate["frozen_backbone"])
    return model


def train_mode(model: nn.Module, frozen: bool) -> None:
    model.train()
    if frozen:
        # Freezing parameters alone does not freeze BatchNorm running statistics.
        model.features.eval()


def amp_context(device: torch.device, enabled: bool):
    return (
        torch.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda" and enabled
        else contextlib.nullcontext()
    )


def validation_probability(
    model: nn.Module, images: KneeImages, config: dict, device: torch.device
) -> np.ndarray:
    require(
        images.partition.name == "VALIDATION" and not images.augment,
        "Only deterministic VALIDATION scoring allowed",
    )
    images.partition.validate()
    model.eval()
    p, seen = np.empty(len(images), dtype=np.float64), []
    with torch.inference_mode():
        for x, _, index in loader(images, config, shuffle=False):
            probability = model(x.to(device)).flatten().sigmoid().cpu().numpy()
            require(len(probability) == len(index), "Output shape mismatch")
            p[index.numpy()] = probability
            seen.extend(index.tolist())
    require(sorted(seen) == list(range(len(images))), "Missing/duplicate validation inference")
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "Invalid image probability")
    return p


def selection_key(metrics: dict, index: int = 0) -> tuple:
    return (-metrics["AUROC"], -metrics["AUPRC"], metrics["Brier"], metrics["log_loss"], index)


class EarlyStop:
    def __init__(self, minimum: int, patience: int):
        self.minimum, self.patience = minimum, patience
        self.best_auc, self.bad, self.best_epoch, self.best_key = -np.inf, 0, 0, None

    def observe(self, epoch: int, score: dict) -> tuple[bool, bool]:
        key = selection_key(score)
        save = self.best_key is None or key < self.best_key
        if save:
            self.best_key, self.best_epoch = key, epoch
        if score["AUROC"] > self.best_auc:
            self.best_auc, self.bad = score["AUROC"], 0
        elif epoch >= self.minimum:
            self.bad += 1
        return save, epoch >= self.minimum and self.bad >= self.patience


def atomic_checkpoint(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="checkpoint_", suffix=".pt", dir=path.parent)
    os.close(fd)
    temp = Path(temporary)
    try:
        torch.save(value, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def fit_candidate(
    data: ImageData,
    config: dict,
    candidate: dict,
    weights: Path,
    device: torch.device,
    output: Path,
    provenance: dict,
    *,
    model_factory=build_model,
) -> dict:
    data.validate()
    set_seed(config["seed"])
    train = KneeImages(data.development.train, data.root, config, augment=True)
    valid = KneeImages(data.development.validation, data.root, config, augment=False)
    batches = loader(train, config, shuffle=True)
    model = model_factory(candidate, weights).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=candidate["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["cuda_amp"])
    criterion, stopping = (
        nn.BCEWithLogitsLoss(),
        EarlyStop(config["minimum_epochs"], config["patience"]),
    )
    checkpoint = output / f"candidate_{candidate['index']}/best.pt"
    history = []
    for epoch in range(1, candidate["epochs"] + 1):
        train.epoch = epoch
        train_mode(model, candidate["frozen_backbone"])
        loss_sum = 0.0
        for x, y, _ in batches:
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, config["cuda_amp"]):
                logits = model(x.to(device)).flatten()
                loss = criterion(logits, y.to(device))
            require(torch.isfinite(loss).item(), "Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(y)
        probability = validation_probability(model, valid, config, device)
        score = tabular.metrics(data.development.validation, probability)
        history.append({"epoch": epoch, "TRAIN_BCE": loss_sum / len(train), "VALIDATION": score})
        save, stop = stopping.observe(epoch, score)
        if save:
            atomic_checkpoint(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "candidate": candidate,
                    "provenance": provenance,
                    "validation_metrics": score,
                },
                checkpoint,
            )
        if stop:
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(
        saved["epoch"] == stopping.best_epoch and saved["provenance"] == provenance,
        "Best-checkpoint provenance/epoch mismatch",
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    probability = validation_probability(model, valid, config, device)
    score = tabular.metrics(data.development.validation, probability)
    require(
        selection_key(score) == selection_key(saved["validation_metrics"]),
        "Checkpoint metrics do not reproduce",
    )
    publish_json(history, output / f"candidate_{candidate['index']}/history.json")
    print(
        f"Completed image candidate {candidate['index']}, best epoch {stopping.best_epoch}",
        flush=True,
    )
    return {
        "candidate": candidate,
        "best_epoch": stopping.best_epoch,
        "epochs_run": len(history),
        "early_stopped": len(history) < candidate["epochs"],
        "metrics": score,
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "probability": probability,
    }


def provenance(
    data: ImageData, config: dict, config_path: Path, weights: Path, device: torch.device
) -> dict:
    preprocessing, augmentation = specifications(config)
    return {
        "source_hashes": data.development.source_hashes,
        "model_configuration_sha256": sha256_file(config_path),
        "preprocessing_sha256": digest(preprocessing),
        "augmentation_sha256": digest(augmentation),
        "training_code_sha256": sha256_file(Path(__file__)),
        "normalization_source_sha256": sha256_file(Path("src/imaging/preprocessing.py")),
        "validation_metric_code_sha256": sha256_file(Path(tabular.__file__)),
        "pretrained_identifier": config["weights"],
        "pretrained_sha256": sha256_file(weights),
        "seed": config["seed"],
        "bootstrap_seed": config["bootstrap_seed"],
        "DataLoader_seed": config["seed"],
        "augmentation_seed": config["seed"],
        "TRAIN_count": len(data.development.train.frame),
        "VALIDATION_count": len(data.development.validation.frame),
        "device": device_info(device),
    }


def preserve(baseline: dict) -> None:
    tabular.preserve(baseline["previous_protected"])
    for group in ("tabular_files", "tabular_code_config"):
        require(
            {p: sha256_file(p) for p in baseline[group]} == baseline[group],
            "Approved tabular artifact/code changed",
        )
    require(
        set(baseline["tabular_files"])
        == {str(p) for p in Path("data/processed/modeling/tabular").rglob("*") if p.is_file()},
        "Tabular artifact set changed",
    )


def plot_calibration(
    partition: tabular.DevelopmentPartition, probability: np.ndarray, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from sklearn.calibration import calibration_curve

    actual, predicted = calibration_curve(
        partition.labels(), probability, n_bins=10, strategy="quantile"
    )
    publish_json(
        {
            "mean_probability": predicted.tolist(),
            "event_fraction": actual.tolist(),
            "partition": "VALIDATION",
            "recalibration_applied": False,
        },
        output / "calibration_data.json",
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(predicted, actual, "o-")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean probability",
        ylabel="Observed fraction",
        title="VALIDATION image calibration",
    )
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=150)
    plt.close(fig)


def production(config_path: Path, output: Path, device: torch.device) -> dict:
    config = load_config(config_path)
    validate_config(config)
    weights = pretrained_path(output)
    data = load_image_data()
    meta = provenance(data, config, config_path, weights, device)
    frozen_path = output / "run_freeze.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        require(
            frozen["provenance"] == meta and frozen["TEST_scored"] is False,
            "Frozen image run provenance changed",
        )
        for name, value in frozen["artifacts_sha256"].items():
            require(sha256_file(output / name) == value, "Frozen image artifact SHA mismatch")
        integrity(data)
        selected = read_json(output / "selected_candidate.json")
        state = torch.load(output / selected["checkpoint"], map_location="cpu", weights_only=True)
        require(state["provenance"] == meta, "Selected checkpoint provenance mismatch")
        model = build_model(selected["candidate"], weights).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        probability = validation_probability(
            model,
            KneeImages(data.development.validation, data.root, config, augment=False),
            config,
            device,
        )
        predictions = pd.read_parquet(output / "validation_predictions.parquet")
        tabular.validate_predictions(
            data.development.validation, predictions, {"DenseNet121_selected"}
        )
        np.testing.assert_array_equal(predictions.predicted_probability.to_numpy(), probability)
        return frozen
    require(
        not list(output.glob("candidate_*/best.pt")),
        "Partial production checkpoints; refusing overwrite. Use an explicit fresh run directory.",
    )
    publish_json(config, output / "image_model_v1.yaml")
    for name, spec in zip(
        ("preprocessing_spec", "augmentation_spec"), specifications(config), strict=True
    ):
        publish_json({"specification": spec, "provenance": meta}, output / f"{name}.json")
    publish_json(
        {"provenance": meta, "config": config, "declared_before_training": True},
        output / "predeclared_run.json",
    )
    baseline = read_json(OUTPUT / "pre_change_preservation.json")
    preserve(baseline)
    results = []
    for candidate in config["candidates"]:
        # Repeat frozen source AND every development artifact verification before every fit.
        current = load_image_data()
        require(
            provenance(current, config, config_path, weights, device) == meta,
            "Fit source/configuration changed",
        )
        audit = integrity(current)
        publish_json(
            {"provenance": meta, "audit": audit},
            output / f"candidate_{candidate['index']}/input_integrity.json",
        )
        results.append(fit_candidate(current, config, candidate, weights, device, output, meta))
    best = min(results, key=lambda r: selection_key(r["metrics"], r["candidate"]["index"]))
    probability = best["probability"]
    predictions = data.development.validation.frame[[*tabular.IDENTITY, tabular.TARGET]].copy()
    predictions["predicted_probability"] = probability
    predictions["model_id"] = pd.array(["DenseNet121_selected"] * len(predictions), dtype="string")
    predictions["split"] = pd.array(["VALIDATION"] * len(predictions), dtype="string")
    tabular.validate_predictions(data.development.validation, predictions, {"DenseNet121_selected"})
    publish_parquet(predictions, output / "validation_predictions.parquet")
    publish_json(
        {
            "provenance": meta,
            "candidates": [{k: v for k, v in r.items() if k != "probability"} for r in results],
        },
        output / "candidate_results.json",
    )
    publish_json(
        {**{k: v for k, v in best.items() if k != "probability"}, "provenance": meta},
        output / "selected_candidate.json",
    )
    publish_json(
        {"provenance": meta, "metrics": best["metrics"]}, output / "validation_metrics.json"
    )
    boot_config = {
        "bootstrap_replicates": config["bootstrap_replicates"],
        "bootstrap_seed": config["bootstrap_seed"],
    }
    boot, intervals = tabular.bootstrap(
        data.development.validation, {"DenseNet121_selected": probability}, boot_config
    )
    publish_parquet(boot, output / "bootstrap_metrics.parquet")
    publish_json({"provenance": meta, **intervals}, output / "clustered_bootstrap.json")
    plot_calibration(data.development.validation, probability, output)
    preserve(baseline)
    catalog = {
        str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in ("run_freeze.json", "run_freeze.json.lock")
    }
    frozen = {
        "status": "FROZEN_IMAGE_DEVELOPMENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": meta,
        "artifacts_sha256": catalog,
        "TEST_scored": False,
        "TEST_images_loaded_by_model_loader": 0,
        "default_target": tabular.TARGET,
        "not_final_project_model": True,
        "selected_checkpoint": best["checkpoint"],
        "source_preservation_passed": True,
    }
    publish_json(frozen, frozen_path)
    return frozen


def sanity_and_compute(
    data: ImageData, config: dict, weights: Path, device: torch.device, output: Path
) -> dict:
    set_seed(config["seed"])
    part = data.development.train
    # Balanced tiny TRAIN sample, not a scientific endpoint evaluation or candidate selection.
    index = np.concatenate([np.flatnonzero(part.labels() == value)[:4] for value in (0, 1)])
    require(len(index) == 8, "Tiny TRAIN sanity requires four images per class")
    images = KneeImages(part, data.root, config, augment=False)
    x = torch.stack([images[int(i)][0] for i in index]).to(device)
    y = torch.tensor(part.labels()[index], dtype=torch.float32, device=device)
    model = build_model(config["candidates"][0], weights).to(device)
    model.eval()
    with torch.inference_mode():
        features = model.features(x)
        features = (
            nn.functional.adaptive_avg_pool2d(nn.functional.relu(features), (1, 1))
            .flatten(1)
            .clone()
        )
    # Do not retain an inference tensor in an autograd graph.
    features = features.detach().clone()
    head = nn.Linear(features.shape[1], 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    first = float(loss_fn(head(features).flatten(), y).detach().cpu())
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(head(features).flatten(), y)
        loss.backward()
        opt.step()
    final = float(loss_fn(head(features).flatten(), y).detach().cpu())
    require(
        final < first * 0.5
        and ((head(features).flatten() > 0) == y.bool()).float().mean().item() > 0.5,
        "Tiny TRAIN overfit sanity failed",
    )
    montage(data, config, index[:6], output / "TRAIN_transform_montage.png")
    del model, features, x, head, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Actual architecture/resolution/batch and full gradients, no validation or TEST images.
    model = build_model(config["candidates"][1], weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["cuda_amp"])
    batch = torch.stack([images[int(i)][0] for i in index[: config["batch_size"]]]).to(device)
    labels = y[: len(batch)]
    times = []
    for step in range(config["local_compute_gate"]["benchmark_batches"] + 1):
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        start = time.perf_counter()
        with amp_context(device, config["cuda_amp"]):
            loss = loss_fn(model(batch).flatten(), labels)
        scaler.scale(loss).backward()
        require(
            any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.features.parameters()
            ),
            "Backbone gradient sanity failed",
        )
        scaler.step(optimizer)
        scaler.update()
        sync(device)
        if step:
            times.append(time.perf_counter() - start)
    per_image = float(np.median(times)) / len(batch)
    # Conservative maximum-epoch estimate treats reference/validation passes as full train steps;
    # decode/percentile I/O is excluded, so actual wall time can be longer.
    projected = (
        per_image
        * sum(c["epochs"] for c in config["candidates"])
        * (len(part.frame) + len(data.development.validation.frame))
        / 3600
    )
    return {
        "implementation_sanity_passed": True,
        "scientific_metrics": False,
        "tiny_subset_partition": "TRAIN",
        "tiny_subset_images": 8,
        "loss_decreased": final < first,
        "backbone_gradients_finite": True,
        "device": device_info(device),
        "batch_size": len(batch),
        "input_size": 320,
        "median_full_training_batch_seconds": float(np.median(times)),
        "projected_maximum_epoch_compute_hours": projected,
        "projection_caveat": "Full-train equivalent conservative estimate, excludes image decoding/I/O and warm-up; not an exact runtime prediction.",
        "local_training_practical": projected
        <= config["local_compute_gate"]["maximum_projected_hours"],
        "predeclared_local_hour_limit": config["local_compute_gate"]["maximum_projected_hours"],
    }


def montage(data: ImageData, config: dict, indices: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    recipe = WEIGHTS.transforms()
    mean, std = torch.tensor(recipe.mean)[:, None, None], torch.tensor(recipe.std)[:, None, None]
    fig, axes = plt.subplots(len(indices), 2, figsize=(7, 3 * len(indices)), layout="constrained")
    for number, index in enumerate(indices):
        row = data.development.train.frame.iloc[int(index)]
        array = np.load(
            resolve_image(data.root, row.effective_crop_relative_path), allow_pickle=False
        )
        for column, augmented in enumerate((False, True)):
            image = transform_image(
                array, config, train=augmented, key=tuple(row[c] for c in tabular.IDENTITY), epoch=1
            )
            view = (image * std + mean)[0].clamp(0, 1)
            axes[number, column].imshow(view, cmap="gray", vmin=0, vmax=1)
            axes[number, column].set_title(
                f"TRAIN sample {number + 1}: {'affine' if augmented else 'deterministic'}"
            )
            axes[number, column].axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def prepare_readiness(
    data: ImageData, config: dict, weights: Path, device: torch.device, output: Path, meta: dict
) -> dict:
    """Reuse measured readiness; timings are observations, not deterministic rebuild targets."""
    path = output / "readiness_audit.json"
    montage_path = output / "TRAIN_transform_montage.png"
    if path.exists():
        recorded = read_json(path)
        require(recorded["provenance"] == meta, "Readiness provenance changed")
        require(
            sha256_file(montage_path) == recorded["montage_sha256"],
            "Readiness montage changed",
        )
        require(recorded["implementation_sanity_passed"] is True, "Readiness sanity failed")
        return recorded
    recorded = {
        "provenance": meta,
        **sanity_and_compute(data, config, weights, device, output),
        "montage_sha256": sha256_file(montage_path),
    }
    publish_json(recorded, path)
    return recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run production candidates only after TRAIN sanity passes",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = load_config(CONFIG)
    validate_config(config)
    device = device_for(args.device)
    torch.set_num_threads(2)
    with file_lock(args.output / "run_freeze.json"):
        if args.train and (args.output / "run_freeze.json").exists():
            result = production(CONFIG, args.output, device)
        else:
            baseline = read_json(OUTPUT / "pre_change_preservation.json")
            preserve(baseline)
            data = load_image_data()
            audit = integrity(data)
            weights = pretrained_path(args.output)
            meta = provenance(data, config, CONFIG, weights, device)
            publish_json(config, args.output / "image_model_v1.yaml")
            for name, spec in zip(
                ("preprocessing_spec", "augmentation_spec"), specifications(config), strict=True
            ):
                publish_json(
                    {"specification": spec, "provenance": meta}, args.output / f"{name}.json"
                )
            publish_json(
                {"provenance": meta, "config": config, "declared_before_training": True},
                args.output / "predeclared_run.json",
            )
            publish_json(
                {"provenance": meta, "audit": audit}, args.output / "input_integrity_audit.json"
            )
            readiness = prepare_readiness(data, config, weights, device, args.output, meta)
            preserve(baseline)
            if args.train and (device.type == "cuda" or readiness["local_training_practical"]):
                result = production(CONFIG, args.output, device)
            else:
                result = {
                    "status": "PORTABLE_CODE_READY_PRODUCTION_NOT_RUN",
                    "readiness": readiness,
                    "TEST_scored": False,
                }
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    from modeling.image import main as canonical_main

    raise SystemExit(canonical_main())
