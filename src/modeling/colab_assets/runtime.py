"""Filesystem-only adapter for the immutable TRAIN/VALIDATION execution bundle."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path

import pandas as pd

from imaging.artifact_io import sha256_file
from modeling import tabular
from multimodal.freeze import read_json, require

VERSION = "colab_t4_v1"
COUNTS = {"TRAIN": (4873, 749, 2535), "VALIDATION": (1044, 161, 543)}
SOURCES = {
    "dataset": "382d17228ca6eb85c064f4a2c2d8c402a2479bde6fc746c9a55f7fbd57eac190",
    "participants": "f2c4f4b0e3260f9a9f482acb88d5f53fdcad61bf66261f3d947ad3b7da3115ba",
    "knees": "4658b04a667a747c9bbc5422fa5ec7f516677b98714e46e3bbba7fc974502815",
    "feature_groups": "cde2f2dbb4eba356559daf4c854e3f955bac99daa5cf9c03afd2f6052bde9e28",
    "dataset_freeze": "293e628798661b71bd1145433a7df4c45b425dfcd38f1c661853e5117756461e",
    "split_freeze": "af0fe879044aa68bbafba7cbd71aee1fee26612775f49cfbd4f803c381ec9dbf",
    "schema": "92dae260d28083789a796337ea12762de3d2e09779666ba5747ff8ce6f64b19e",
    "imaging_manifest": "cce476f5ecc2381232b026a9aec09dffe58db391116deadb7fddb2405aee0fa7",
}
CODE = {
    "src/modeling/image.py": "5a47d6032d441974e065f5b1d477a4bbe31bfe0f8de6477c916794e90dd31793",
    "configs/image_model_v1.yaml": "895331f848144ff2582de626345476800a887065972d17903a2209c80acdefbb",
    "src/imaging/preprocessing.py": "fd23712784c43ebb2f72835d0e0f5718cb93cc76d657ffe639d287c547815edb",
}
METRIC_PARENT_SHA = "5da741f438100547b88fde285d74f815fe5f9cd25981eba8672d5c18485c9c06"
WEIGHT_SHA = "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395"
PRODUCTION = Path("data/processed/modeling/image/v1/production_cuda")
MANIFEST_COLUMNS = [
    *tabular.IDENTITY,
    tabular.TARGET,
    "effective_crop_relative_path",
    "effective_crop_sha256",
    "effective_provenance",
    "anatomical_side",
    "split",
]


def safe_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Invalid bundle path")
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, "Non-relative bundle path")
    path = root / p
    require(path.resolve().is_relative_to(root.resolve()), "Bundle path escapes root")
    require(not path.is_symlink() and path.is_file(), "Bundle artifact missing/symlink")
    return path


def blocks(path: Path, names: list[str]) -> dict[str, str]:
    import hashlib

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    selected = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
            selected[node.name] = hashlib.sha256(
                "".join(lines[start - 1 : node.end_lineno]).encode()
            ).hexdigest()
    require(set(selected) == set(names), "Required verbatim source block missing")
    return selected


def tables(root: Path, expected: dict = COUNTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(root / "metadata/image_manifest.parquet")
    people = pd.read_parquet(root / "metadata/participant_split_projection.parquet")
    knees = pd.read_parquet(root / "metadata/knee_split_projection.parquet")
    require(list(manifest) == MANIFEST_COLUMNS, "Unexpected image-loader metadata columns")
    require(
        list(people)
        == ["participant_id", "split", "eligible_knee_count", "primary_event_knee_count"]
        and list(knees) == [*tabular.IDENTITY, "anatomical_side", "split"],
        "Unexpected split projection columns",
    )
    require(not manifest.isna().any().any(), "Missing portable image metadata/label")
    require(
        pd.api.types.is_bool_dtype(manifest[tabular.TARGET]), "Primary label must remain Boolean"
    )
    require(
        not people.isna().any().any() and not knees.isna().any().any(), "Missing split metadata"
    )
    for frame in (manifest, people, knees):
        require(set(frame.split) == set(expected), "TEST/unknown split in execution bundle")
    require(not people.participant_id.duplicated().any(), "Duplicate participant assignment")
    require(not manifest.duplicated(list(tabular.IDENTITY)).any(), "Duplicate image knee")
    require(not knees.duplicated(list(tabular.IDENTITY)).any(), "Duplicate knee assignment")
    require(
        manifest.knee_side_code.isin(["1", "2"]).all() and manifest.baseline_visit.eq("V00").all(),
        "Wrong knee side/baseline visit",
    )
    require(
        manifest.anatomical_side.eq(manifest.knee_side_code.map({"1": "R", "2": "L"})).all(),
        "Wrong anatomical side",
    )
    require(
        manifest.effective_provenance.isin(["AUTO_PASS", "HUMAN_ACCEPT", "HUMAN_OVERRIDE"]).all(),
        "Unresolved image provenance",
    )
    require(
        not manifest.effective_crop_relative_path.duplicated().any(), "Duplicate effective image"
    )
    pd.testing.assert_frame_equal(
        manifest[[*tabular.IDENTITY, "anatomical_side", "split"]], knees, check_exact=True
    )
    require(
        set(manifest.participant_id) == set(people.participant_id), "Participant linkage mismatch"
    )
    membership = people.set_index("participant_id").split.to_dict()
    require(manifest.participant_id.map(membership).eq(manifest.split).all(), "Participant leakage")
    observed = manifest.groupby("participant_id", sort=True).agg(
        eligible_knee_count=(tabular.TARGET, "size"),
        primary_event_knee_count=(tabular.TARGET, "sum"),
    )
    comparison = people.set_index("participant_id").loc[observed.index]
    for c in observed:
        require(observed[c].eq(comparison[c]).all(), "Participant knee/event census mismatch")
    require(observed.eligible_knee_count.isin([1, 2]).all(), "Unexpected knee contribution")
    for role, (n, events, participants) in expected.items():
        rows = manifest.loc[manifest.split.eq(role)]
        require(
            (len(rows), int(rows[tabular.TARGET].sum()), rows.participant_id.nunique())
            == (n, events, participants),
            "Portable TRAIN/VALIDATION census mismatch",
        )
    return manifest, people, knees


def verify(root: Path, expected: dict = COUNTS) -> dict:
    root = root.resolve()
    freeze = read_json(root / "bundle_freeze.json")
    require(freeze["version"] == VERSION, "Wrong bundle version")
    require(freeze["source_hashes"] == SOURCES, "Frozen original source expectations changed")
    require(freeze["approved_code_hashes"] == CODE, "Approved code/config expectations changed")
    for name, value in CODE.items():
        require(sha256_file(safe_path(root, name)) == value, "Approved code/config changed")
    catalog = freeze["artifacts_sha256"]
    present = set()
    for p in root.rglob("*"):
        require(not p.is_symlink(), "Symlinks prohibited in execution bundle")
        if p.is_file() and p.suffix.lower() == ".npy":
            require(
                p.is_relative_to(root / "images"), "Image outside the approved execution image tree"
            )
        if p.is_file() and not p.is_relative_to(root / PRODUCTION):
            present.add(p.relative_to(root).as_posix())
    require(present == set(catalog) | {"bundle_freeze.json"}, "Unexpected/missing bundle artifact")
    for name, value in catalog.items():
        require(sha256_file(safe_path(root, name)) == value, "Bundle artifact SHA mismatch")
    lineage = freeze["verbatim_metrics"]
    require(lineage["parent_sha256"] == METRIC_PARENT_SHA, "Metrics parent source changed")
    require(
        blocks(root / "src/modeling/tabular.py", list(lineage["blocks_sha256"]))
        == lineage["blocks_sha256"],
        "Scientific metric/bootstrap helper changed",
    )
    require(
        read_json(root / "source_expectations.json") == SOURCES,
        "Original frozen source expectations mismatch",
    )
    manifest, _, _ = tables(root, expected)
    wanted = {"images/" + p for p in manifest.effective_crop_relative_path}
    actual = {p.relative_to(root).as_posix() for p in (root / "images").rglob("*") if p.is_file()}
    require(actual == wanted, "Unexpected/missing image; TEST images prohibited")
    require({n for n in catalog if n.startswith("images/")} == wanted, "Image catalog mismatch")
    for row in manifest.itertuples(index=False):
        path = safe_path(root, "images/" + row.effective_crop_relative_path)
        require(
            sha256_file(path) == row.effective_crop_sha256, "Packaged effective image SHA mismatch"
        )
    require(
        sha256_file(safe_path(root, "pretrained/densenet121-a639ec97.pth")) == WEIGHT_SHA,
        "Official pretrained weights changed",
    )
    return {
        "TRAIN": expected["TRAIN"][0],
        "VALIDATION": expected["VALIDATION"][0],
        "TEST": 0,
        "images": len(wanted),
        "bundle_freeze_sha256": sha256_file(root / "bundle_freeze.json"),
        "all_artifact_and_image_SHA_checks_passed": True,
    }


def approved_gpu_priority(name: str) -> int | None:
    """Hardware-only allowlist; lower priority prefers A100, then L4, then T4."""
    import re

    match = re.match(r"^(?:NVIDIA|Tesla)\s+(A100|L4|T4)\b", name, re.IGNORECASE)
    return ("A100", "L4", "T4").index(match[1].upper()) if match else None


def environment(root: Path, *, cuda: bool) -> dict:
    pins = read_json(root / "dependency_versions.json")
    actual = {name: importlib.metadata.version(name) for name in pins}
    for name, value in actual.items():
        base, _, build = value.partition("+")
        require(
            base == pins[name]
            and (not build or name in ("torch", "torchvision") and build.startswith("cu")),
            f"Pinned dependency incompatible: {name} expected {pins[name]}, observed {value}; STOP",
        )
    import torch
    import torchvision

    require(torch.__version__.split("+")[0] == pins["torch"], "PyTorch runtime version mismatch")
    require(
        torchvision.__version__.split("+")[0] == pins["torchvision"], "torchvision runtime mismatch"
    )
    report = {
        "distribution_versions": actual,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
    }
    if cuda:
        require(
            torch.cuda.is_available(),
            "CUDA unavailable/incompatible driver; STOP; do not substitute versions",
        )
        approved = []
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            priority = approved_gpu_priority(name)
            if priority is not None:
                approved.append((priority, index, name))
        require(bool(approved), "Approved CUDA GPU required: NVIDIA A100, L4 or T4; STOP")
        _, index, name = min(approved)
        torch.cuda.set_device(index)
        report.update({"GPU": name, "CUDA_device_index": index, "CUDA_runtime": torch.version.cuda})
        torch.ones(1, device="cuda")
        torch.cuda.synchronize()
    return report


def load_data(root: Path):
    from modeling.image import ImageData

    verify(root)
    manifest, people, _ = tables(root)
    membership = people.set_index("participant_id").split.to_dict()
    parts = {}
    for role in ("TRAIN", "VALIDATION"):
        rows = manifest.loc[manifest.split.eq(role)].drop(columns="split").reset_index(drop=True)
        parts[role] = tabular.DevelopmentPartition(
            role,
            rows,
            membership,
            frozenset(rows[list(tabular.IDENTITY)].itertuples(index=False, name=None)),
        )
    data = ImageData(
        tabular.DevelopmentData(parts["TRAIN"], parts["VALIDATION"], dict(SOURCES)), root / "images"
    )
    data.validate()
    return data


def bind(root: Path) -> None:
    """Change I/O boundaries only; the approved training functions are untouched."""
    from modeling import image

    os.chdir(root)
    verify(root)
    image.OUTPUT = root / "data/processed/modeling/image/v1"
    image.CONFIG = root / "configs/image_model_v1.yaml"
    image.IMAGE_ROOT = root / "images"
    image.load_image_data = lambda: load_data(root)

    def preserve(baseline):
        require(
            baseline == {"version": VERSION, "source_hashes": SOURCES},
            "Portable preservation baseline changed",
        )
        verify(root)

    image.preserve = preserve
    original_provenance = image.provenance

    def provenance(*args, **kwargs):
        result = original_provenance(*args, **kwargs)
        freeze = read_json(root / "bundle_freeze.json")
        result["portable_execution"] = {
            "version": VERSION,
            "bundle_freeze_sha256": sha256_file(root / "bundle_freeze.json"),
            "derived_manifest_sha256": freeze["artifacts_sha256"][
                "metadata/image_manifest.parquet"
            ],
            "original_metrics_source_sha256": METRIC_PARENT_SHA,
            "adapter_sha256": sha256_file(Path(__file__)),
            "preservation_scope": "Execution bundle; full original sources verified on Mac at packaging/import, not shipped",
        }
        return result

    image.provenance = provenance
    original_pretrained = image.pretrained_path

    def pretrained(output):
        import shutil

        target = output / "pretrained/densenet121-a639ec97.pth"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "pretrained/densenet121-a639ec97.pth", target)
        require(sha256_file(target) == WEIGHT_SHA, "Production pretrained weights mismatch")
        return original_pretrained(output)

    image.pretrained_path = pretrained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    result = verify(args.bundle)
    if args.cuda:
        result["runtime"] = environment(args.bundle, cuda=True)
    print(json.dumps(result, indent=2))
    return 0
