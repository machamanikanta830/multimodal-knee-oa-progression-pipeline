"""Immutability baseline for the frozen V3 automatic crops during manual adjudication.

Manual adjudication runs alongside the automatic dataset for as long as human review takes, so the
frozen automatic crops need a hash baseline recorded before the first decision. The first call
creates the ledger; every later call verifies it and reports any artifact that was added, removed,
or modified.

This exists so a later finalization step can state, with evidence rather than assumption, that the
human-adjudicated manifest was built on the same pixels the automatic pass produced, and so a
corrected crop written into the separate ``overrides`` tree can be shown not to have disturbed the
automatic crop it supersedes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from imaging.artifact_io import atomic_json, atomic_parquet, hash_tree

DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/v3_frozen_full")
AUTOMATIC_CROP_LEDGER = Path("audit/automatic_crop_ledger.parquet")
AUTOMATIC_CROP_SUMMARY = Path("audit/automatic_crop_preservation.json")
PANEL_FILE_SUFFIX = "_uint16.npy"


class AdjudicationAuditError(ValueError):
    """Raised when the automatic crop dataset is not in the state adjudication assumes."""


def _compare(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, list[str]]:
    before = previous.set_index("artifact")
    after = current.set_index("artifact")
    shared = sorted(set(after.index) & set(before.index))
    return {
        "added": sorted(set(after.index) - set(before.index)),
        "removed": sorted(set(before.index) - set(after.index)),
        "modified": [
            artifact
            for artifact in shared
            if after.loc[artifact, "sha256"] != before.loc[artifact, "sha256"]
            or int(after.loc[artifact, "bytes"]) != int(before.loc[artifact, "bytes"])
        ],
    }


def audit_automatic_crops(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = 8,
    strict: bool = False,
) -> dict[str, Any]:
    """Create or verify the hash ledger covering every automatic crop and completion record.

    Passing ``strict`` turns a detected change into an error, which is how a finalization step
    should call it. The default reports the difference so an operator can inspect it first.
    """

    output_root = Path(output_root)
    crops_root = output_root / "crops"
    if not crops_root.is_dir():
        raise AdjudicationAuditError("The frozen V3 automatic crop tree does not exist")
    current = hash_tree(crops_root, relative_to=output_root, workers=workers)
    panels = int(current["artifact"].str.endswith(PANEL_FILE_SUFFIX).sum())
    records = int(current["artifact"].str.endswith("record.json").sum())

    ledger_path = output_root / AUTOMATIC_CROP_LEDGER
    if ledger_path.is_file():
        difference = _compare(pd.read_parquet(ledger_path), current)
        action = "ledger_verified"
    else:
        atomic_parquet(current, ledger_path)
        difference = {"added": [], "removed": [], "modified": []}
        action = "ledger_created"

    unchanged = not any(difference.values())
    summary = {
        "action": action,
        "files": len(current),
        "bytes": int(current["bytes"].sum()) if len(current) else 0,
        "panel_crops": panels,
        "completion_records": records,
        "all_unchanged": unchanged,
        "added": len(difference["added"]),
        "removed": len(difference["removed"]),
        "modified": len(difference["modified"]),
        # A short sample keeps the summary readable while still naming what moved.
        "changed_examples": sorted(
            difference["added"] + difference["removed"] + difference["modified"]
        )[:10],
        "ledger_relative_path": AUTOMATIC_CROP_LEDGER.as_posix(),
    }
    atomic_json(summary, output_root / AUTOMATIC_CROP_SUMMARY)
    if strict and not unchanged:
        raise AdjudicationAuditError("The frozen V3 automatic crops changed since the baseline")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record or verify the hash baseline that proves the frozen V3 automatic crops are "
            "unchanged while manual adjudication proceeds."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any automatic crop changed since the baseline.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_automatic_crops(
        output_root=args.dataset_root, workers=args.workers, strict=args.strict
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
