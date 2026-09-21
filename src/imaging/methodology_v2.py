"""Outcome-blind methodology sampling and preservation audits for imaging V2.

All manifests produced by this module contain protected linkage fields and therefore belong only
under Git-ignored local data directories. The selections are imaging-methodology samples, not
modeling train/validation/test partitions.
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

DEVELOPMENT_SIZE = 192
HOLDOUT_REPRESENTATIVE_SIZE = 128
HOLDOUT_STRESS_SIZE = 128
LATERALITY_AUDIT_SIZE = 128
MIN_REPORTABLE_GROUP = 5

DEFAULT_V1_ROOT = Path("data/processed/oai_images/full_v1")
DEFAULT_V1_ACQUISITIONS = Path(
    "data/processed/manifests/v00_xray_full_preprocessing_acquisitions.parquet"
)
DEFAULT_V1_PANELS = Path("data/processed/manifests/v00_xray_full_standardized_images.parquet")
DEFAULT_V1_REVIEW = DEFAULT_V1_ROOT / "qc/visual_review_manifest.parquet"
DEFAULT_PILOT = Path("data/processed/manifests/v00_xray_pilot_manifest.parquet")
DEFAULT_VALIDATION = Path("data/processed/manifests/v00_xray_validation128_manifest.parquet")
DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/v2_candidate")
DEFAULT_V1_LEDGER = DEFAULT_OUTPUT_ROOT / "audit/v1_file_hashes.parquet"
DEFAULT_HOLDOUT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests/v2_holdout_manifest.parquet"
DEFAULT_LATERALITY_AUDIT = DEFAULT_OUTPUT_ROOT / "manifests/v2_laterality_audit_manifest.parquet"


class MethodologyV2Error(ValueError):
    """Raised when a preservation or sample-selection invariant fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _v1_files(
    v1_root: Path, acquisition_manifest_path: Path, panel_manifest_path: Path
) -> list[tuple[Path, str]]:
    files = [path for path in v1_root.rglob("*") if path.is_file()]
    files.extend((acquisition_manifest_path, panel_manifest_path))
    if not all(path.is_file() for path in files):
        raise MethodologyV2Error("A required V1 artifact is missing")
    result = []
    for path in sorted(set(files)):
        if path.is_relative_to(v1_root):
            label = f"full_v1/{path.relative_to(v1_root).as_posix()}"
        else:
            label = f"external_manifest/{path.name}"
        result.append((path, label))
    return result


def capture_v1_hashes(
    *,
    v1_root: Path = DEFAULT_V1_ROOT,
    acquisition_manifest_path: Path = DEFAULT_V1_ACQUISITIONS,
    panel_manifest_path: Path = DEFAULT_V1_PANELS,
    ledger_path: Path = DEFAULT_V1_LEDGER,
) -> dict[str, Any]:
    """Create a local immutable-content ledger before any V2 artifact is generated."""

    if ledger_path.exists():
        raise MethodologyV2Error("V1 preservation ledger already exists; use verification")
    files = _v1_files(v1_root, acquisition_manifest_path, panel_manifest_path)

    def record(item: tuple[Path, str]) -> dict[str, Any]:
        path, label = item
        return {"artifact": label, "bytes": path.stat().st_size, "sha256": _sha256(path)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(record, files))
    frame = pd.DataFrame(rows).sort_values("artifact", kind="stable", ignore_index=True)
    _atomic_parquet(frame, ledger_path)
    summary = {
        "preprocessing_version": "v1_provisional",
        "files_hashed": len(frame),
        "bytes_hashed": int(frame["bytes"].sum()),
        "duplicate_content_hashes": int(frame["sha256"].duplicated().sum()),
    }
    _atomic_json(summary, ledger_path.with_name("v1_preservation_summary.json"))
    return summary


def verify_v1_hashes(
    *,
    v1_root: Path = DEFAULT_V1_ROOT,
    acquisition_manifest_path: Path = DEFAULT_V1_ACQUISITIONS,
    panel_manifest_path: Path = DEFAULT_V1_PANELS,
    ledger_path: Path = DEFAULT_V1_LEDGER,
) -> dict[str, Any]:
    """Verify that every V1 provisional artifact remains byte-identical."""

    ledger = pd.read_parquet(ledger_path).set_index("artifact")
    files = _v1_files(v1_root, acquisition_manifest_path, panel_manifest_path)
    current = {label: path for path, label in files}
    if set(current) != set(ledger.index):
        raise MethodologyV2Error("The V1 artifact inventory changed")

    def matches(label: str) -> bool:
        path = current[label]
        row = ledger.loc[label]
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


def _feature_key(row: Any, column: str) -> str:
    value = getattr(row, column)
    if pd.isna(value) or str(value).strip() == "":
        return "<missing>"
    return str(value)


def _coverage_select(
    frame: pd.DataFrame,
    count: int,
    *,
    seed: str,
    priority_padding: int = 0,
) -> list[int]:
    if count > len(frame):
        raise MethodologyV2Error("Not enough candidates for the requested methodology sample")
    features = (
        "manufacturer",
        "manufacturer_model_name",
        "original_row_spacing_mm",
        "dimension_family",
        "image_release_study",
        "xray_accept_qc",
        "qc_problem_signature",
        "eligible_knee_count",
    )
    candidates = frame.copy()
    candidates["stable_hash"] = candidates["acquisition_index"].map(
        lambda value: _stable_hash(int(value), seed)
    )
    selected: list[int] = []
    if priority_padding:
        top = candidates.sort_values(
            ["maximum_padding_mm", "stable_hash"], ascending=[False, True], kind="stable"
        ).head(min(priority_padding, count))
        selected.extend(top["acquisition_index"].astype(int).tolist())

    covered: dict[tuple[str, str], int] = {}
    for index in selected:
        row = candidates.loc[candidates["acquisition_index"].eq(index)].iloc[0]
        for column in features:
            key = (column, "<missing>" if pd.isna(row[column]) else str(row[column]))
            covered[key] = covered.get(key, 0) + 1
    while len(selected) < count:
        available = candidates.loc[~candidates["acquisition_index"].isin(selected)]
        best_index: int | None = None
        best_score: tuple[float, str] | None = None
        for row in available.itertuples(index=False):
            novelty = 0.0
            for column in features:
                key = (column, _feature_key(row, column))
                novelty += 1.0 / (1.0 + covered.get(key, 0))
            score = (-novelty, row.stable_hash)
            if best_score is None or score < best_score:
                best_score = score
                best_index = int(row.acquisition_index)
        if best_index is None:
            raise MethodologyV2Error("Coverage selection terminated early")
        selected.append(best_index)
        row = candidates.loc[candidates["acquisition_index"].eq(best_index)].iloc[0]
        for column in features:
            key = (column, "<missing>" if pd.isna(row[column]) else str(row[column]))
            covered[key] = covered.get(key, 0) + 1
    return selected


def _methodology_frame(
    acquisitions: pd.DataFrame, panels: pd.DataFrame, v1_review: pd.DataFrame
) -> pd.DataFrame:
    maximum_padding = panels.groupby("acquisition_index")[
        ["horizontal_padding_mm", "vertical_padding_mm"]
    ].max()
    maximum_padding["maximum_padding_mm"] = maximum_padding.max(axis=1)
    frame = acquisitions.merge(
        maximum_padding[["maximum_padding_mm"]],
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
    flags = [
        "xray_alignment_problem",
        "xray_centering_problem",
        "xray_incomplete_depiction",
        "xray_positioning_problem",
    ]
    frame["qc_problem_signature"] = frame[flags].fillna("").astype(str).agg("|".join, axis=1)
    labels = v1_review.set_index("acquisition_index")[
        ["localization_visual_state", "crop_visual_state"]
    ]
    frame = frame.merge(labels, left_on="acquisition_index", right_index=True, how="left")
    return frame


def select_methodology_samples(
    *,
    acquisition_manifest_path: Path = DEFAULT_V1_ACQUISITIONS,
    panel_manifest_path: Path = DEFAULT_V1_PANELS,
    v1_review_path: Path = DEFAULT_V1_REVIEW,
    pilot_manifest_path: Path = DEFAULT_PILOT,
    validation_manifest_path: Path = DEFAULT_VALIDATION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Create development and sealed holdout samples without reading OA outcomes."""

    acquisitions = pd.read_parquet(acquisition_manifest_path)
    panels = pd.read_parquet(panel_manifest_path)
    review = pd.read_parquet(v1_review_path)
    frame = _methodology_frame(acquisitions, panels, review)
    reference_to_index = dict(
        zip(frame["associated_file_reference"], frame["acquisition_index"], strict=True)
    )
    pilot_references = pd.read_parquet(pilot_manifest_path, columns=["image_file"])["image_file"]
    validation_references = pd.read_parquet(validation_manifest_path, columns=["image_file"])[
        "image_file"
    ]
    pilot_indices = {int(reference_to_index[value]) for value in pilot_references}
    validation_indices = {int(reference_to_index[value]) for value in validation_references}
    reviewed_indices = set(review["acquisition_index"].astype(int))

    known_failures = set(
        review.loc[review["localization_visual_state"].eq("FAILED"), "acquisition_index"].astype(
            int
        )
    )
    known_borderline = set(
        review.loc[
            review["localization_visual_state"].eq("BORDERLINE"), "acquisition_index"
        ].astype(int)
    )
    mandatory_development = reviewed_indices | pilot_indices
    if len(mandatory_development) > DEVELOPMENT_SIZE:
        raise MethodologyV2Error("Mandatory development cases exceed the target size")
    validation_pool = frame.loc[
        frame["acquisition_index"].isin(validation_indices - mandatory_development)
    ]
    additions = _coverage_select(
        validation_pool,
        DEVELOPMENT_SIZE - len(mandatory_development),
        seed="oai-v2-development-v1",
    )
    development_indices = sorted(mandatory_development | set(additions))

    previously_inspected = reviewed_indices | pilot_indices | validation_indices
    unseen = frame.loc[
        ~frame["acquisition_index"].isin(previously_inspected | set(development_indices))
    ]
    representative = (
        unseen.assign(
            stable_hash=unseen["acquisition_index"].map(
                lambda value: _stable_hash(int(value), "oai-v2-holdout-representative-v1")
            )
        )
        .sort_values("stable_hash", kind="stable")
        .head(HOLDOUT_REPRESENTATIVE_SIZE)
    )
    representative_indices = set(representative["acquisition_index"].astype(int))
    stress_pool = unseen.loc[~unseen["acquisition_index"].isin(representative_indices)]
    stress_indices = set(
        _coverage_select(
            stress_pool,
            HOLDOUT_STRESS_SIZE,
            seed="oai-v2-holdout-stress-v1",
            priority_padding=32,
        )
    )
    holdout_indices = representative_indices | stress_indices

    if len(development_indices) != DEVELOPMENT_SIZE or len(holdout_indices) != 256:
        raise MethodologyV2Error("Methodology sample sizes are incorrect")
    if set(development_indices).intersection(holdout_indices):
        raise MethodologyV2Error("Development and holdout acquisitions overlap")
    if holdout_indices.intersection(previously_inspected):
        raise MethodologyV2Error("Independent holdout contains a previously inspected acquisition")
    if not known_failures.issubset(development_indices) or not known_borderline.issubset(
        development_indices
    ):
        raise MethodologyV2Error("Known failures or borderline cases are absent from development")

    development = frame.loc[frame["acquisition_index"].isin(development_indices)].copy()
    development["methodology_role"] = "V2_DEVELOPMENT"
    development["selection_outcomes_read"] = False
    holdout = frame.loc[frame["acquisition_index"].isin(holdout_indices)].copy()
    holdout["methodology_role"] = holdout["acquisition_index"].map(
        lambda value: (
            "V2_HOLDOUT_REPRESENTATIVE"
            if int(value) in representative_indices
            else "V2_HOLDOUT_STRESS"
        )
    )
    holdout["selection_outcomes_read"] = False
    development = development.sort_values("acquisition_index", kind="stable", ignore_index=True)
    holdout = holdout.sort_values(["methodology_role", "acquisition_index"], ignore_index=True)

    manifest_root = output_root / "manifests"
    _atomic_parquet(development, manifest_root / "v2_development_manifest.parquet")
    _atomic_parquet(holdout, manifest_root / "v2_holdout_manifest.parquet")
    audit = {
        "development_acquisitions": len(development),
        "development_known_failures": int(
            development["localization_visual_state"].eq("FAILED").sum()
        ),
        "development_known_borderline": int(
            development["localization_visual_state"].eq("BORDERLINE").sum()
        ),
        "development_known_successes": int(
            development["localization_visual_state"].eq("SUCCESS").sum()
        ),
        "holdout_acquisitions": len(holdout),
        "holdout_representative": int(
            holdout["methodology_role"].eq("V2_HOLDOUT_REPRESENTATIVE").sum()
        ),
        "holdout_stress": int(holdout["methodology_role"].eq("V2_HOLDOUT_STRESS").sum()),
        "development_holdout_overlap": 0,
        "holdout_previously_inspected_overlap": 0,
        "selection_outcomes_read": False,
        "modeling_split_created": False,
    }
    _atomic_json(audit, manifest_root / "selection_summary.json")
    return audit


def select_laterality_audit_sample(
    *,
    holdout_manifest_path: Path = DEFAULT_HOLDOUT_MANIFEST,
    output_path: Path = DEFAULT_LATERALITY_AUDIT,
    count: int = LATERALITY_AUDIT_SIZE,
) -> dict[str, Any]:
    """Select an outcome-blind, heterogeneity-focused laterality audit sample.

    This is an imaging-methodology audit subset of the sealed V2 holdout, not a modeling split.
    Protected linkage columns are retained only in the local Git-ignored manifest. Every observed
    manufacturer, model, spacing, release, and QC category is covered when the requested sample
    has sufficient capacity; exact dimension families represented by fewer than five holdout
    acquisitions are treated as non-reportable singletons rather than disclosure strata.
    """

    holdout = pd.read_parquet(holdout_manifest_path)
    if holdout["acquisition_index"].duplicated().any():
        raise MethodologyV2Error("Holdout manifest has duplicate acquisitions")
    if holdout.get("selection_outcomes_read", pd.Series([True])).astype(bool).any():
        raise MethodologyV2Error("Laterality audit source is not outcome-blind")
    selected_indices = _coverage_select(
        holdout,
        count,
        seed="oai-v2-laterality-audit-v1",
    )
    selected = holdout.loc[holdout["acquisition_index"].isin(selected_indices)].copy()
    selected["laterality_audit_role"] = "INDEPENDENT_MANUAL_AUDIT"
    selected["laterality_outcomes_read"] = False
    selected = selected.sort_values("acquisition_index", kind="stable", ignore_index=True)
    if len(selected) != count:
        raise MethodologyV2Error("Laterality audit size is incorrect")

    coverage_columns = (
        "manufacturer",
        "manufacturer_model_name",
        "original_row_spacing_mm",
        "image_release_study",
        "xray_accept_qc",
        "qc_problem_signature",
    )
    missing_categories: dict[str, list[str]] = {}
    for column in coverage_columns:
        available = set(holdout[column].fillna("<missing>").astype(str))
        observed = set(selected[column].fillna("<missing>").astype(str))
        missing_categories[column] = sorted(available - observed)
    reportable_dimensions = set(
        holdout["dimension_family"]
        .value_counts()
        .loc[lambda values: values >= MIN_REPORTABLE_GROUP]
        .index
    )
    selected_dimensions = set(selected["dimension_family"].astype(str))
    missing_categories["reportable_dimension_family"] = sorted(
        reportable_dimensions - selected_dimensions
    )
    if any(missing_categories.values()):
        raise MethodologyV2Error(
            "Laterality audit does not cover every required heterogeneity category"
        )

    _atomic_parquet(selected, output_path)
    summary = {
        "audit_acquisitions": len(selected),
        "source_holdout_acquisitions": len(holdout),
        "unique_acquisitions": int(selected["acquisition_index"].nunique()),
        "manufacturer_categories": int(selected["manufacturer"].fillna("<missing>").nunique()),
        "scanner_model_categories": int(
            selected["manufacturer_model_name"].fillna("<missing>").nunique()
        ),
        "spacing_families": int(selected["original_row_spacing_mm"].nunique(dropna=False)),
        "reportable_dimension_families_covered": len(reportable_dimensions),
        "image_releases": int(selected["image_release_study"].nunique(dropna=False)),
        "qc_categories": int(selected["xray_accept_qc"].nunique(dropna=False)),
        "selection_outcomes_read": False,
        "modeling_split_created": False,
    }
    _atomic_json(summary, output_path.with_suffix(".summary.json"))
    return summary
