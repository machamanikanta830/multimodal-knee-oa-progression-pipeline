"""Freeze one deterministic participant-grouped split of the approved multimodal dataset.

Search sees only participant identity, eligible knee count and primary-event knee count. All
other balance variables are descriptive after selection. No preprocessing or modeling occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from datetime import UTC, datetime
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from imaging.artifact_io import file_lock
from multimodal.freeze import (
    CLINICAL,
    DEFAULT_DIRECTORY,
    DEFAULT_IMAGE_ROOT,
    FUNCTION,
    PRO,
    load_config,
    load_frozen_dataset,
    outcome_counts,
    preservation_snapshot,
    publish_json,
    publish_parquet,
    read_json,
    require,
    sha256_file,
)

DEFAULT_OUTPUT = DEFAULT_DIRECTORY / "splits/v1"
DEFAULT_CONFIG = Path("configs/split_v1.yaml")
APPROVED_DATASET_SHA = "382d17228ca6eb85c064f4a2c2d8c402a2479bde6fc746c9a55f7fbd57eac190"
SPLITS = ("TRAIN", "VALIDATION", "TEST")
KEYS = ("participant_id", "knee_side_code", "baseline_visit")
KNEE_IDENTITY = (*KEYS, "knee_side_label", "anatomical_side")
ARTIFACTS = (
    "participant_split_manifest.parquet",
    "knee_split_manifest.parquet",
    "split_summary.json",
    "split_balance_audit.json",
)
OBJECTIVE = (
    "participant_overlap_hard_failure",
    "participant_share_max_then_sum_absolute_error",
    "knee_share_max_then_sum_absolute_error",
    "primary_prevalence_max_then_sum_absolute_error",
    "primary_event_share_max_then_sum_absolute_error",
    "one_two_knee_participant_share_max_then_sum_absolute_error",
    "lowest_candidate_index_tie_break",
)
SPARSE_POLICY = (
    "eligible_knees_and_event_knees",
    "eligible_knees_and_any_event",
    "any_event",
    "all_participants",
)


def validate_config(config: dict) -> None:
    require(config["split_names"] == list(SPLITS), "Unexpected split names")
    require(config["proportion_weights"] == [70, 15, 15], "Unexpected split proportions")
    require(config["objective"] == list(OBJECTIVE), "Unapproved search objective")
    require(config["sparse_strata_policy"] == list(SPARSE_POLICY), "Sparse-strata policy changed")
    require(
        config["selection_columns"]
        == ["participant_id", "knee_side_code", "composite_progression"],
        "Predictors/secondary outcomes must not enter split search",
    )
    require(
        config["algorithm_version"] == "sha256_stratified_controlled_rounding_search_v1",
        "Unknown split-generation algorithm",
    )
    require(isinstance(config["seed"], str) and bool(config["seed"]), "Missing fixed seed")
    require(
        isinstance(config["candidate_count"], int) and 1 <= config["candidate_count"] <= 4096,
        "Invalid deterministic search size",
    )
    require(config["minimum_stratum_participants"] >= 20, "Unsafe sparse-stratum threshold")


def validate_rows(frame: pd.DataFrame) -> None:
    require(not frame[list(KEYS)].isna().any().any(), "Missing participant/knee identity")
    require(
        not frame.duplicated(["participant_id", "knee_side_code"]).any(), "Duplicate source knee"
    )
    require(frame.baseline_visit.eq("V00").all(), "Unexpected source visit")
    require(frame.knee_side_code.isin(["1", "2"]).all(), "Invalid source knee side")
    require(
        frame.knee_side_label.eq(frame.knee_side_code.map({"1": "right", "2": "left"})).all()
        and frame.anatomical_side.eq(frame.knee_side_code.map({"1": "R", "2": "L"})).all(),
        "Wrong source anatomical side",
    )
    require(
        not frame.composite_progression.isna().any()
        and str(frame.composite_progression.dtype) in ("bool", "boolean"),
        "Primary outcome must be a complete Boolean label",
    )
    require(
        frame.groupby("participant_id").size().isin([1, 2]).all(),
        "Participant must contribute one or two eligible knees",
    )


def participant_strata(frame: pd.DataFrame, minimum: int) -> tuple[pd.DataFrame, dict]:
    """Conservatively coarsen globally only if a stratum cannot support stable holdouts."""
    validate_rows(frame)
    table = (
        frame.groupby("participant_id", sort=True)
        .agg(
            eligible_knee_count=("knee_side_code", "size"),
            primary_event_knee_count=("composite_progression", "sum"),
        )
        .reset_index()
    )
    table["participant_id"] = table.participant_id.astype(frame.participant_id.dtype)
    for c in ("eligible_knee_count", "primary_event_knee_count"):
        table[c] = table[c].astype("int64")
    require(len(table) >= 5, "Too few participants for three nonempty partitions")
    original = {
        f"K{k}_E{e}": int(n)
        for (k, e), n in table.groupby(["eligible_knee_count", "primary_event_knee_count"])
        .size()
        .items()
    }
    attempts = []
    for stage in SPARSE_POLICY:
        if stage == "eligible_knees_and_event_knees":
            labels = [
                f"K{k}_E{e}"
                for k, e in zip(
                    table.eligible_knee_count, table.primary_event_knee_count, strict=True
                )
            ]
        elif stage == "eligible_knees_and_any_event":
            labels = [
                f"K{k}_ANY{int(e > 0)}"
                for k, e in zip(
                    table.eligible_knee_count, table.primary_event_knee_count, strict=True
                )
            ]
        elif stage == "any_event":
            labels = [f"ANY{int(e > 0)}" for e in table.primary_event_knee_count]
        else:
            labels = ["ALL"] * len(table)
        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        attempts.append({"stage": stage, "counts": {k: int(v) for k, v in counts.items()}})
        if min(counts.values()) >= minimum or stage == "all_participants":
            table["stratification_group"] = pd.array(labels, dtype="string")
            break
    return table, {
        "original_strata": original,
        "selected_stage": stage,
        "selected_strata": attempts[-1]["counts"],
        "minimum_participants": minimum,
        "collapsed": stage != SPARSE_POLICY[0],
        "attempts": attempts,
        "reason": "No sparse strata; preserve knee count and zero/one/two event burden."
        if stage == SPARSE_POLICY[0]
        else "Sparse strata coarsened using only knee count/primary burden; no predictor information.",
    }


def quotas(n: int, weights: list[int]) -> tuple[int, ...]:
    base = [n * w // 100 for w in weights]
    order = sorted(range(3), key=lambda i: (-(n * weights[i] % 100), i))
    for i in order[: n - sum(base)]:
        base[i] += 1
    return tuple(base)


def controlled_allocations(sizes: list[int], weights: list[int]) -> list[tuple]:
    """Enumerate floor/ceil stratum allocations satisfying exact global Hamilton quotas."""
    target = quotas(sum(sizes), weights)
    base = [tuple(n * w // 100 for w in weights) for n in sizes]
    deficits = tuple(target[i] - sum(row[i] for row in base) for i in range(3))
    options = []
    for n, row in zip(sizes, base, strict=True):
        flexible = [i for i, w in enumerate(weights) if n * w % 100]
        options.append(list(itertools.combinations(flexible, n - sum(row))))
    matrices = []

    def visit(j, left, chosen):
        if j == len(sizes):
            if left == (0, 0, 0):
                matrices.append(tuple(chosen))
            return
        for increment in options[j]:
            remaining = tuple(left[i] - int(i in increment) for i in range(3))
            if min(remaining) >= 0:
                visit(
                    j + 1,
                    remaining,
                    [*chosen, tuple(base[j][i] + int(i in increment) for i in range(3))],
                )

    visit(0, deficits, [])
    require(bool(matrices), "No feasible controlled rounding allocation")
    return matrices


def main_counts(rows: list[tuple], assignments: dict) -> dict:
    counts = {
        s: {"participants": 0, "knees": 0, "events": 0, "one_knee": 0, "two_knees": 0}
        for s in SPLITS
    }
    require(len(assignments) == len(rows), "Missing/duplicate participant assignment")
    for participant, knees, events, _ in rows:
        c = counts[assignments[participant]]
        c["participants"] += 1
        c["knees"] += knees
        c["events"] += events
        c["one_knee" if knees == 1 else "two_knees"] += 1
    require(all(c["participants"] > 0 for c in counts.values()), "Empty split")
    return counts


def objective(counts: dict, weights: list[int]) -> tuple[Fraction, ...]:
    """Exact rational, lexicographic objective fixed before candidate comparison."""
    totals = {k: sum(c[k] for c in counts.values()) for k in next(iter(counts.values()))}
    proportions = [Fraction(w, 100) for w in weights]
    score = [Fraction(0)]  # Overlap is a construction invariant and subsequently audited.
    for metric in ("participants", "knees", "prevalence", "events", "knee_contribution"):
        errors = []
        for s, p in zip(SPLITS, proportions, strict=True):
            c = counts[s]
            if metric == "prevalence":
                errors.append(
                    abs(
                        Fraction(c["events"], c["knees"])
                        - Fraction(totals["events"], totals["knees"])
                    )
                )
            elif metric == "knee_contribution":
                errors.extend(
                    abs(Fraction(c[k], totals[k]) - p)
                    for k in ("one_knee", "two_knees")
                    if totals[k]
                )
            elif totals[metric]:
                errors.append(abs(Fraction(c[metric], totals[metric]) - p))
        score.extend((max(errors, default=Fraction(0)), sum(errors, Fraction(0))))
    return tuple(score)


def generate(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validate_config(config)
    participants, strata = participant_strata(frame, config["minimum_stratum_participants"])
    rows = list(participants.itertuples(index=False, name=None))
    groups = {
        g: [r for r in rows if r[3] == g]
        for g in sorted(participants.stratification_group.unique())
    }
    matrices = controlled_allocations(
        [len(v) for v in groups.values()], config["proportion_weights"]
    )
    best = None
    candidates = []
    for candidate in range(config["candidate_count"]):
        matrix_index = candidate % len(matrices)
        assignment = {}
        for (group, members), allocation in zip(
            groups.items(), matrices[matrix_index], strict=True
        ):
            ranked = sorted(
                members,
                key=lambda row: (
                    hashlib.sha256(
                        f"{config['seed']}|{candidate}|{group}|{row[0]}".encode()
                    ).digest(),
                    row[0],
                ),
            )
            offset = 0
            for split, n in zip(SPLITS, allocation, strict=True):
                for row in ranked[offset : offset + n]:
                    require(row[0] not in assignment, "Participant overlap during generation")
                    assignment[row[0]] = split
                offset += n
            require(offset == len(members), "Stratum participant dropped")
        counts = main_counts(rows, assignment)
        score = objective(counts, config["proportion_weights"])
        candidates.append(
            {
                "candidate_index": candidate,
                "allocation_index": matrix_index,
                "objective_exact": [str(x) for x in score],
            }
        )
        if best is None or (score, candidate) < (best[0], best[1]):
            best = score, candidate, assignment, counts
    score, candidate, assignment, counts = best
    participants["split"] = pd.array(participants.participant_id.map(assignment), dtype="string")
    participants["split_generation_version"] = pd.array(
        [config["split_version"]] * len(participants), dtype="string"
    )
    participants["split_seed"] = pd.array([config["seed"]] * len(participants), dtype="string")
    participants["candidate_index"] = candidate
    knees = frame[list(KNEE_IDENTITY)].copy().reset_index(drop=True)
    knees["split"] = pd.array(knees.participant_id.map(assignment), dtype="string")
    knees["dataset_row_index"] = range(len(knees))
    validate_manifests(frame, participants, knees, config)
    return (
        participants,
        knees,
        {
            "stratification": strata,
            "participant_quotas": dict(
                zip(SPLITS, quotas(len(participants), config["proportion_weights"]), strict=True)
            ),
            "candidate_count": config["candidate_count"],
            "feasible_allocation_count": len(matrices),
            "feasible_allocations_searched": min(len(matrices), config["candidate_count"]),
            "selected_candidate_index": candidate,
            "objective_definition": list(OBJECTIVE),
            "objective_exact": [str(x) for x in score],
            "objective_numeric": [float(x) for x in score],
            "candidates": candidates,
            "selection_columns": config["selection_columns"],
            "predictors_and_secondary_descriptors_used_for_selection": False,
            "main_counts": counts,
        },
    )


def validate_manifests(
    frame: pd.DataFrame, participants: pd.DataFrame, knees: pd.DataFrame, config: dict
) -> dict:
    validate_rows(frame)
    require(
        not participants.participant_id.isna().any()
        and not participants.participant_id.duplicated().any(),
        "Duplicate/null participant assignment",
    )
    require(
        set(participants.participant_id) == set(frame.participant_id), "Missing/extra participant"
    )
    require(participants.split.isin(SPLITS).all(), "Unknown/null participant split")
    require(
        not knees[list(KEYS)].isna().any().any() and not knees.duplicated(list(KEYS)).any(),
        "Duplicate/null knee assignment",
    )
    require(len(knees) == len(frame), "Missing/extra knee row")
    pd.testing.assert_frame_equal(
        knees[list(KNEE_IDENTITY)],
        frame[list(KNEE_IDENTITY)].reset_index(drop=True),
        check_exact=True,
    )
    require(
        knees.dataset_row_index.tolist() == list(range(len(frame))), "Knee row ordering changed"
    )
    assignment = participants.set_index("participant_id").split
    require(
        knees.split.isin(SPLITS).all()
        and knees.split.eq(knees.participant_id.map(assignment)).all(),
        "Participant overlap or wrong knee/participant mapping",
    )
    require(knees.groupby("participant_id").split.nunique().eq(1).all(), "Participant leakage")
    expected, _ = participant_strata(frame, config["minimum_stratum_participants"])
    actual = participants[
        [
            "participant_id",
            "eligible_knee_count",
            "primary_event_knee_count",
            "stratification_group",
        ]
    ]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_exact=True)
    require(
        participants.split_generation_version.eq(config["split_version"]).all()
        and participants.split_seed.eq(config["seed"]).all(),
        "Generation metadata mismatch",
    )
    require(
        participants.candidate_index.nunique() == 1
        and 0 <= int(participants.candidate_index.iloc[0]) < config["candidate_count"],
        "Invalid candidate metadata",
    )
    sets = {s: set(participants.loc[participants.split.eq(s), "participant_id"]) for s in SPLITS}
    require(
        tuple(len(sets[s]) for s in SPLITS)
        == quotas(len(participants), config["proportion_weights"]),
        "Participant quota mismatch",
    )
    overlaps = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in itertools.combinations(SPLITS, 2)}
    require(
        not any(overlaps.values()) and set.union(*sets.values()) == set(frame.participant_id),
        "Participant overlap or union mismatch",
    )
    return {
        "participant_pairwise_overlap": overlaps,
        "participants_assigned_once": len(participants),
        "knees_assigned_once": len(knees),
        "missing_participants": 0,
        "extra_participants": 0,
        "missing_knees": 0,
        "extra_knees": 0,
        "duplicate_knees": 0,
        "silent_drops": 0,
        "both_knees_share_split": True,
    }


def category_counts(series: pd.Series, levels: tuple) -> dict:
    return {str(v): int(series.eq(v).sum()) for v in levels}


def balance_audit(
    frame: pd.DataFrame, participants: pd.DataFrame, knees: pd.DataFrame, config: dict
) -> dict:
    integrity = validate_manifests(frame, participants, knees, config)
    total_events = int(frame.composite_progression.sum())
    global_prevalence = total_events / len(frame)
    totals = {
        "participants": len(participants),
        "knees": len(frame),
        "events": total_events,
        "one_knee": int(participants.eligible_knee_count.eq(1).sum()),
        "two_knees": int(participants.eligible_knee_count.eq(2).sum()),
    }
    result, deviations, flags = {}, {}, []
    for s, weight in zip(SPLITS, config["proportion_weights"], strict=True):
        rows = frame.loc[knees.split.eq(s).to_numpy()]
        p = participants.loc[participants.split.eq(s)]
        events = int(rows.composite_progression.sum())
        modality_masks = {
            "clinical_complete": rows[list(CLINICAL)].notna().all(axis=1),
            "patient_reported_complete": rows[list(PRO)].notna().all(axis=1),
            "physical_function_complete": rows[list(FUNCTION)].notna().all(axis=1),
            "imaging_complete": rows.effective_crop_relative_path.notna()
            & rows.effective_crop_sha256.notna(),
        }
        availability = {name: int(mask.sum()) for name, mask in modality_masks.items()}
        availability["all_four_modalities_complete"] = int(
            pd.concat(modality_masks.values(), axis=1).all(axis=1).sum()
        )
        counts = {
            "participants": len(p),
            "knees": len(rows),
            "events": events,
            "one_knee": int(p.eligible_knee_count.eq(1).sum()),
            "two_knees": int(p.eligible_knee_count.eq(2).sum()),
        }
        prevalence = events / len(rows)
        result[s] = {
            "participants": len(p),
            "participant_percentage": 100 * len(p) / len(participants),
            "knees": len(rows),
            "knee_percentage": 100 * len(rows) / len(frame),
            "primary_endpoint": {
                "positive": events,
                "negative": len(rows) - events,
                "prevalence": prevalence,
            },
            "participant_event_burden": category_counts(p.primary_event_knee_count, (0, 1, 2)),
            "participant_knee_contribution": category_counts(p.eligible_knee_count, (1, 2)),
            "side": category_counts(rows.knee_side_label, ("left", "right")),
            "baseline_KL": {
                **category_counts(rows.baseline_kl, (0, 1, 2, 3)),
                "missing": int(rows.baseline_kl.isna().sum()),
            },
            "image_provenance": category_counts(
                rows.effective_provenance, ("AUTO_PASS", "HUMAN_ACCEPT", "HUMAN_OVERRIDE")
            ),
            "modality_availability": availability,
            "secondary_outcomes_descriptive_only": {
                k: v for k, v in outcome_counts(rows).items() if k != "composite_progression"
            },
        }
        target = weight / 100
        descriptors = {}
        for metric, count in counts.items():
            n = totals[metric]
            share = count / n if n else None
            descriptors[metric] = {
                "actual_count": count,
                "target_count": target * n,
                "absolute_count_difference": abs(count - target * n),
                "share_of_total": share,
                "target_share": target,
                "absolute_share_difference": abs(share - target) if n else None,
                "relative_to_target_count": count / (target * n) if n else None,
            }
        descriptors["primary_prevalence"] = {
            "actual": prevalence,
            "overall": global_prevalence,
            "absolute_difference": abs(prevalence - global_prevalence),
            "relative_to_overall": prevalence / global_prevalence if global_prevalence else None,
        }
        deviations[s] = descriptors
        for metric, threshold_name, quantum in (
            ("participants", "participant_share_absolute_difference", 1),
            ("knees", "knee_share_absolute_difference", 2),
            ("events", "primary_event_share_absolute_difference", 1),
            ("one_knee", "one_two_knee_participant_share_absolute_difference", 1),
            ("two_knees", "one_two_knee_participant_share_absolute_difference", 1),
        ):
            n = totals[metric]
            limit = (
                max(config["audit_material_imbalance_thresholds"][threshold_name], quantum / n)
                if n
                else None
            )
            if n and descriptors[metric]["absolute_share_difference"] > limit + 1e-12:
                flags.append(
                    {
                        "split": s,
                        "metric": metric,
                        "absolute_difference": descriptors[metric]["absolute_share_difference"],
                        "threshold": limit,
                    }
                )
        limit = max(
            config["audit_material_imbalance_thresholds"]["primary_prevalence_absolute_difference"],
            1 / len(rows),
        )
        if abs(prevalence - global_prevalence) > limit + 1e-12:
            flags.append(
                {
                    "split": s,
                    "metric": "primary_prevalence",
                    "absolute_difference": abs(prevalence - global_prevalence),
                    "threshold": limit,
                }
            )
    prevalences = [v["primary_endpoint"]["prevalence"] for v in result.values()]
    return {
        "splits": result,
        "main_balance_deviations": deviations,
        "overall_primary_prevalence": global_prevalence,
        "maximum_pairwise_primary_prevalence_difference": max(prevalences) - min(prevalences),
        "material_imbalance_flags": flags,
        "thresholds": config["audit_material_imbalance_thresholds"],
        "discreteness_policy": "Flag thresholds allow at least one participant/event or two knees of integer rounding, and one event per split for prevalence.",
        "descriptive_variables_not_optimized": [
            "side",
            "baseline_KL",
            "image_provenance",
            "modality_availability",
            "secondary_outcomes",
        ],
        "integrity": integrity,
        "no_preprocessing_or_modeling_performed": True,
    }


def verify_source(directory: Path, approved_sha: str, expected: tuple | None) -> pd.DataFrame:
    require(
        sha256_file(directory / "final_multimodal_dataset.parquet") == approved_sha,
        "Changed frozen dataset SHA",
    )
    frame = load_frozen_dataset(directory)
    validate_rows(frame)
    if expected is not None:
        require(
            (len(frame), frame.participant_id.nunique(), len(frame.columns)) == expected,
            "Unexpected source row/participant/column count",
        )
    return frame


def source_files(directory: Path) -> dict:
    return {
        str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def verify_preservation(baseline: dict, source: Path) -> None:
    require(
        source_files(source) == baseline["multimodal_files"], "Frozen multimodal bundle changed"
    )
    preservation_snapshot(baseline["protected_original"], DEFAULT_IMAGE_ROOT)


def _build_split(
    *,
    source: Path = DEFAULT_DIRECTORY,
    output: Path = DEFAULT_OUTPUT,
    config_path: Path = DEFAULT_CONFIG,
    approved_sha: str = APPROVED_DATASET_SHA,
    expected: tuple | None = (6961, 3621, 54),
    preservation: dict | None = None,
) -> dict:
    source, output, config_path = map(Path, (source, output, config_path))
    config = load_config(config_path)
    validate_config(config)
    before = source_files(source)
    if preservation is not None:
        print("Verifying protected pre-change fingerprints...", flush=True)
        verify_preservation(preservation, source)
    frame = verify_source(source, approved_sha, expected)
    if (output / "split_freeze.json").exists():
        load_frozen_split(output, source=source, approved_sha=approved_sha, expected=expected)
    print("Searching primary-only deterministic participant candidates...", flush=True)
    participants, knees, search = generate(frame, config)
    audit = balance_audit(frame, participants, knees, config)
    summary = {
        "split_version": config["split_version"],
        "configuration": config,
        "source_dataset_sha256": approved_sha,
        "source_rows": len(frame),
        "source_participants": int(frame.participant_id.nunique()),
        "search": search,
        "splits": audit["splits"],
        "shared_by": [
            "formulation_A",
            "formulation_B",
            "tabular",
            "image_only",
            "multimodal",
            "ablations",
            "calibration",
            "interpretability",
        ],
        "test_policy": "Final locked evaluation only; no development/model-performance inspection.",
        "train_policy": "Model fitting and fitting all training-time preprocessing.",
        "validation_policy": "Model/hyperparameter/threshold selection, early stopping and scientifically appropriate calibration-model selection.",
        "model_specific_eligibility_policy": "Retain frozen split labels; never resplit, including complete-case sensitivities.",
        "integrity": audit["integrity"],
    }
    if preservation is not None:
        print(
            "Rechecking frozen dataset, adjudications, images, methodology and raw data...",
            flush=True,
        )
        verify_preservation(preservation, source)
    require(source_files(source) == before, "Source dataset bundle changed during split build")
    # Preflight every existing output before writing anything: never partially replace a freeze.
    for name, value in zip(ARTIFACTS, (participants, knees, summary, audit), strict=True):
        path = output / name
        if path.exists():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(pd.read_parquet(path), value, check_exact=True)
            else:
                require(
                    read_json(path) == value, "Existing frozen split differs; refusing overwrite"
                )
    publish_parquet(participants, output / ARTIFACTS[0])
    publish_parquet(knees, output / ARTIFACTS[1])
    publish_json(summary, output / ARTIFACTS[2])
    publish_json(audit, output / ARTIFACTS[3])
    marker = output / "split_freeze.json"
    existing = read_json(marker) if marker.exists() else None
    frozen = {
        "status": "FROZEN_AUTHORITATIVE",
        "split_version": config["split_version"],
        "source_dataset_path": str(source / "final_multimodal_dataset.parquet"),
        "source_dataset_sha256": approved_sha,
        "source_row_count": len(frame),
        "source_participant_count": int(frame.participant_id.nunique()),
        "source_dataset_freeze_sha256": before[str(source / "dataset_freeze.json")]["sha256"],
        "seed": config["seed"],
        "generation_algorithm_version": config["algorithm_version"],
        "selected_candidate_index": search["selected_candidate_index"],
        "configuration": config,
        "artifacts_sha256": {name: sha256_file(output / name) for name in ARTIFACTS},
        "participant_manifest_sha256": sha256_file(output / ARTIFACTS[0]),
        "knee_manifest_sha256": sha256_file(output / ARTIFACTS[1]),
        "balance_audit_sha256": sha256_file(output / ARTIFACTS[3]),
        "participant_counts": {s: v["participants"] for s, v in audit["splits"].items()},
        "knee_counts": {s: v["knees"] for s, v in audit["splits"].items()},
        "primary_outcomes": {s: v["primary_endpoint"] for s, v in audit["splits"].items()},
        "created_at_utc": existing["created_at_utc"] if existing else datetime.now(UTC).isoformat(),
        "code_configuration_sha256": {
            str(p): sha256_file(p)
            for p in (
                Path(__file__).relative_to(Path.cwd()),
                Path("src/multimodal/freeze.py"),
                Path("src/imaging/artifact_io.py"),
                config_path,
            )
        },
        "runtime": {
            "python": platform.python_version(),
            **{n: version(n) for n in ("pandas", "pyarrow")},
        },
        "sole_authoritative_split": True,
        "all_model_families_and_formulations_share_split": True,
        "future_independent_resplitting_prohibited": True,
        "test_locked_for_final_evaluation_only": True,
        "preprocessing_fitted": False,
        "models_trained": False,
        "source_preservation_passed": True,
        "protected_artifact_preservation_passed": preservation is not None,
        "pre_change_preservation_sha256": sha256_file(output / "pre_change_preservation.json")
        if preservation is not None
        else None,
    }
    publish_json(frozen, marker)
    load_frozen_split(output, source=source, approved_sha=approved_sha, expected=expected)
    return frozen


def build_split(
    *,
    source: Path = DEFAULT_DIRECTORY,
    output: Path = DEFAULT_OUTPUT,
    config_path: Path = DEFAULT_CONFIG,
    approved_sha: str = APPROVED_DATASET_SHA,
    expected: tuple | None = (6961, 3621, 54),
    preservation: dict | None = None,
) -> dict:
    """Serialize all split writers, without locking or writing any protected source artifact."""
    source, output = Path(source), Path(output)
    # Invalid sources must fail before even creating the new split directory/lock sidecar.
    verify_source(source, approved_sha, expected)
    with file_lock(output / "split_freeze.json"):
        return _build_split(
            source=source,
            output=output,
            config_path=config_path,
            approved_sha=approved_sha,
            expected=expected,
            preservation=preservation,
        )


def load_frozen_split(
    output: Path = DEFAULT_OUTPUT,
    *,
    source: Path = DEFAULT_DIRECTORY,
    approved_sha: str = APPROVED_DATASET_SHA,
    expected: tuple | None = (6961, 3621, 54),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify authoritative membership for every future model; never independently resplit."""
    output, source = Path(output), Path(source)
    frozen = read_json(output / "split_freeze.json")
    require(
        frozen["status"] == "FROZEN_AUTHORITATIVE"
        and frozen["source_dataset_sha256"] == approved_sha,
        "Invalid split/source freeze",
    )
    require(
        frozen["sole_authoritative_split"] is True
        and frozen["all_model_families_and_formulations_share_split"] is True
        and frozen["future_independent_resplitting_prohibited"] is True
        and frozen["test_locked_for_final_evaluation_only"] is True
        and frozen["preprocessing_fitted"] is False
        and frozen["models_trained"] is False,
        "Frozen split/model-use policy changed",
    )
    require(set(frozen["artifacts_sha256"]) == set(ARTIFACTS), "Incomplete frozen split artifacts")
    for name, expected_sha in frozen["artifacts_sha256"].items():
        require(sha256_file(output / name) == expected_sha, "Frozen split artifact SHA mismatch")
    for field, artifact in (
        ("participant_manifest_sha256", ARTIFACTS[0]),
        ("knee_manifest_sha256", ARTIFACTS[1]),
        ("balance_audit_sha256", ARTIFACTS[3]),
    ):
        require(
            frozen[field] == frozen["artifacts_sha256"][artifact], "Inconsistent split fingerprints"
        )
    require(
        sha256_file(source / "dataset_freeze.json") == frozen["source_dataset_freeze_sha256"],
        "Source dataset freeze changed",
    )
    config = frozen["configuration"]
    validate_config(config)
    require(
        frozen["split_version"] == config["split_version"]
        and frozen["seed"] == config["seed"]
        and frozen["generation_algorithm_version"] == config["algorithm_version"]
        and frozen["source_dataset_path"] == str(source / "final_multimodal_dataset.parquet"),
        "Frozen split generation/source metadata inconsistent",
    )
    for path, expected_sha in frozen["code_configuration_sha256"].items():
        require(sha256_file(path) == expected_sha, "Frozen split code/configuration changed")
    frame = verify_source(source, approved_sha, expected)
    require(
        len(frame) == frozen["source_row_count"]
        and frame.participant_id.nunique() == frozen["source_participant_count"],
        "Source dimensions changed",
    )
    participants, knees = (
        pd.read_parquet(output / ARTIFACTS[0]),
        pd.read_parquet(output / ARTIFACTS[1]),
    )
    require(
        int(participants.candidate_index.iloc[0]) == frozen["selected_candidate_index"],
        "Frozen selected candidate inconsistent",
    )
    audit = balance_audit(frame, participants, knees, config)
    require(read_json(output / ARTIFACTS[3]) == audit, "Frozen balance audit inconsistent")
    require(
        frozen["participant_counts"] == {s: v["participants"] for s, v in audit["splits"].items()}
        and frozen["knee_counts"] == {s: v["knees"] for s, v in audit["splits"].items()}
        and frozen["primary_outcomes"]
        == {s: v["primary_endpoint"] for s, v in audit["splits"].items()},
        "Frozen split counts inconsistent",
    )
    return participants, knees


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    baseline = read_json(args.output_dir / "pre_change_preservation.json")
    result = build_split(output=args.output_dir, preservation=baseline)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "split_version",
                    "seed",
                    "selected_candidate_index",
                    "participant_manifest_sha256",
                    "knee_manifest_sha256",
                    "participant_counts",
                    "knee_counts",
                    "primary_outcomes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
