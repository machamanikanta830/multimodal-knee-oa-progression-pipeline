"""Outcome-blind V3 methodology sampling and preservation audits.

The manifests produced here contain protected linkage fields and are written only below ignored
local data directories. They are imaging-methodology samples, never modeling partitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from imaging.methodology_v2 import _coverage_select

REPRESENTATIVE_HOLDOUT_SIZE = 192
STRESS_HOLDOUT_SIZE = 192
DEFAULT_ACQUISITIONS = Path(
    "data/processed/manifests/v00_xray_full_preprocessing_acquisitions.parquet"
)
DEFAULT_PANELS = Path("data/processed/manifests/v00_xray_full_standardized_images.parquet")
DEFAULT_PILOT = Path("data/processed/manifests/v00_xray_pilot_manifest.parquet")
DEFAULT_VALIDATION = Path("data/processed/manifests/v00_xray_validation128_manifest.parquet")
DEFAULT_V2_DEVELOPMENT = Path(
    "data/processed/oai_images/v2_candidate/manifests/v2_development_manifest.parquet"
)
DEFAULT_V2_HOLDOUT = Path(
    "data/processed/oai_images/v2_candidate/manifests/v2_holdout_manifest.parquet"
)
DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/localization_v3_candidate")
DEFAULT_V2_ROOT = Path("data/processed/oai_images/v2_candidate")
DEFAULT_V2_LEDGER = DEFAULT_OUTPUT_ROOT / "audit/v2_preservation_ledger.parquet"


class MethodologyV3Error(ValueError):
    """Raised when a V3 selection or preservation invariant fails."""


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=".parquet", dir=target.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".json", dir=target.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_v2_preservation_ledger(
    *,
    v2_root: Path = DEFAULT_V2_ROOT,
    ledger_path: Path = DEFAULT_V2_LEDGER,
) -> dict[str, Any]:
    """Hash every pre-existing V2 artifact before V3 work begins."""

    if ledger_path.exists():
        raise MethodologyV3Error(
            "V2 preservation ledger already exists and will not be overwritten"
        )
    files = sorted(path for path in v2_root.rglob("*") if path.is_file())

    def record(path: Path) -> dict[str, Any]:
        return {
            "artifact": path.relative_to(v2_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(record, files))
    frame = pd.DataFrame(records)
    _atomic_parquet(frame, ledger_path)
    summary = {
        "files": len(frame),
        "bytes": int(frame["bytes"].sum()),
        "duplicate_hashes": int(frame["sha256"].duplicated().sum()),
    }
    _atomic_json(summary, ledger_path.with_suffix(".summary.json"))
    return summary


def verify_v2_preservation(
    *,
    v2_root: Path = DEFAULT_V2_ROOT,
    ledger_path: Path = DEFAULT_V2_LEDGER,
) -> dict[str, Any]:
    """Verify all ledgered V2 artifacts and detect additions/removals under the preserved root."""

    ledger = pd.read_parquet(ledger_path).set_index("artifact")
    current = {
        path.relative_to(v2_root).as_posix(): path for path in v2_root.rglob("*") if path.is_file()
    }
    if set(current) != set(ledger.index):
        raise MethodologyV3Error("The preserved V2 artifact inventory changed")

    def matches(label: str) -> bool:
        row = ledger.loc[label]
        path = current[label]
        return path.stat().st_size == int(row["bytes"]) and _sha256(path) == row["sha256"]

    labels = sorted(current)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(matches, labels))
    return {
        "files_checked": len(results),
        "hash_matches": sum(results),
        "all_unchanged": all(results),
    }


def _stable_hash(value: int, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _base_frame(acquisitions: pd.DataFrame, panels: pd.DataFrame) -> pd.DataFrame:
    padding = panels.groupby("acquisition_index")[
        ["horizontal_padding_mm", "vertical_padding_mm"]
    ].max()
    padding["maximum_padding_mm"] = padding.max(axis=1)
    frame = acquisitions.merge(
        padding[["maximum_padding_mm"]],
        left_on="acquisition_index",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    frame["dimension_family"] = (
        frame["original_rows"].astype(int).astype(str)
        + "x"
        + frame["original_columns"].astype(int).astype(str)
    )
    flags = (
        "xray_alignment_problem",
        "xray_centering_problem",
        "xray_incomplete_depiction",
        "xray_positioning_problem",
    )
    frame["qc_problem_signature"] = frame[list(flags)].fillna("").astype(str).agg("|".join, axis=1)
    return frame


def build_v3_development_manifest(
    *,
    acquisitions_path: Path = DEFAULT_ACQUISITIONS,
    panels_path: Path = DEFAULT_PANELS,
    pilot_path: Path = DEFAULT_PILOT,
    validation_path: Path = DEFAULT_VALIDATION,
    v2_development_path: Path = DEFAULT_V2_DEVELOPMENT,
    v2_holdout_path: Path = DEFAULT_V2_HOLDOUT,
    output_path: Path = DEFAULT_OUTPUT_ROOT / "manifests/v3_development_manifest.parquet",
) -> dict[str, Any]:
    """Combine every previously inspected acquisition into the allowed V3 development pool."""

    if output_path.exists():
        raise MethodologyV3Error(
            "V3 development manifest already exists and will not be overwritten"
        )
    acquisitions = pd.read_parquet(acquisitions_path)
    panels = pd.read_parquet(panels_path)
    frame = _base_frame(acquisitions, panels)
    reference_to_index = dict(
        zip(frame["associated_file_reference"], frame["acquisition_index"], strict=True)
    )
    development_indices = set(
        pd.read_parquet(v2_development_path, columns=["acquisition_index"])[
            "acquisition_index"
        ].astype(int)
    )
    development_indices.update(
        pd.read_parquet(v2_holdout_path, columns=["acquisition_index"])["acquisition_index"].astype(
            int
        )
    )
    for source in (pilot_path, validation_path):
        references = pd.read_parquet(source, columns=["image_file"])["image_file"]
        development_indices.update(int(reference_to_index[value]) for value in references)
    selected = frame.loc[frame["acquisition_index"].isin(development_indices)].copy()
    selected["methodology_role"] = "V3_DEVELOPMENT"
    selected["selection_outcomes_read"] = False
    selected = selected.sort_values("acquisition_index", kind="stable", ignore_index=True)
    if (
        len(selected) != len(development_indices)
        or selected["acquisition_index"].duplicated().any()
    ):
        raise MethodologyV3Error("V3 development linkage is incomplete or duplicated")
    _atomic_parquet(selected, output_path)
    summary = {
        "development_acquisitions": len(selected),
        "unique_acquisitions": int(selected["acquisition_index"].nunique()),
        "selection_outcomes_read": False,
        "modeling_split_created": False,
    }
    _atomic_json(summary, output_path.with_suffix(".summary.json"))
    return summary


def select_v3_holdout(
    *,
    acquisitions_path: Path = DEFAULT_ACQUISITIONS,
    panels_path: Path = DEFAULT_PANELS,
    development_path: Path = DEFAULT_OUTPUT_ROOT / "manifests/v3_development_manifest.parquet",
    output_path: Path = DEFAULT_OUTPUT_ROOT / "manifests/v3_holdout_manifest.parquet",
) -> dict[str, Any]:
    """Seal a new representative/stress holdout disjoint from all prior image review."""

    if output_path.exists():
        raise MethodologyV3Error("V3 holdout manifest already exists and will not be overwritten")
    acquisitions = pd.read_parquet(acquisitions_path)
    panels = pd.read_parquet(panels_path)
    frame = _base_frame(acquisitions, panels)
    development = pd.read_parquet(development_path, columns=["acquisition_index"])
    used = set(development["acquisition_index"].astype(int))
    unseen = frame.loc[~frame["acquisition_index"].isin(used)].copy()
    representative = (
        unseen.assign(
            stable_hash=unseen["acquisition_index"].map(
                lambda value: _stable_hash(int(value), "oai-v3-holdout-representative-v1")
            )
        )
        .sort_values("stable_hash", kind="stable")
        .head(REPRESENTATIVE_HOLDOUT_SIZE)
    )
    representative_indices = set(representative["acquisition_index"].astype(int))
    stress_pool = unseen.loc[~unseen["acquisition_index"].isin(representative_indices)].copy()

    difficult = stress_pool.loc[
        stress_pool["manufacturer"].fillna("").eq("AGFA")
        & stress_pool["manufacturer_model_name"].fillna("").str.startswith("ADC_51")
        & stress_pool["original_row_spacing_mm"].round(3).eq(0.170)
        & stress_pool["dimension_family"].eq("2048x2494")
    ].assign(
        stable_hash=lambda value: value["acquisition_index"].map(
            lambda item: _stable_hash(int(item), "oai-v3-holdout-agfa-stress-v1")
        )
    )
    difficult = difficult.sort_values("stable_hash", kind="stable").head(96)
    difficult_indices = set(difficult["acquisition_index"].astype(int))
    remainder_pool = stress_pool.loc[~stress_pool["acquisition_index"].isin(difficult_indices)]
    remainder_indices = set(
        _coverage_select(
            remainder_pool,
            STRESS_HOLDOUT_SIZE - len(difficult_indices),
            seed="oai-v3-holdout-stress-coverage-v1",
            priority_padding=32,
        )
    )
    stress_indices = difficult_indices | remainder_indices
    holdout_indices = representative_indices | stress_indices
    if len(representative_indices) != 192 or len(stress_indices) != 192:
        raise MethodologyV3Error("V3 holdout components have incorrect sizes")
    if holdout_indices.intersection(used):
        raise MethodologyV3Error("V3 holdout overlaps prior methodology evidence")
    selected = frame.loc[frame["acquisition_index"].isin(holdout_indices)].copy()
    selected["methodology_role"] = selected["acquisition_index"].map(
        lambda value: (
            "V3_HOLDOUT_REPRESENTATIVE"
            if int(value) in representative_indices
            else "V3_HOLDOUT_STRESS"
        )
    )
    selected["selection_outcomes_read"] = False
    selected = selected.sort_values(["methodology_role", "acquisition_index"], ignore_index=True)
    _atomic_parquet(selected, output_path)
    summary = {
        "holdout_acquisitions": len(selected),
        "representative_acquisitions": int(
            selected["methodology_role"].eq("V3_HOLDOUT_REPRESENTATIVE").sum()
        ),
        "stress_acquisitions": int(selected["methodology_role"].eq("V3_HOLDOUT_STRESS").sum()),
        "agfa_adc_51xx_stress_acquisitions": len(difficult_indices),
        "development_overlap": 0,
        "selection_outcomes_read": False,
        "modeling_split_created": False,
    }
    _atomic_json(summary, output_path.with_suffix(".summary.json"))
    return summary
