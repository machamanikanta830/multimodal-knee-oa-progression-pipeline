"""Tests for the automatic-crop immutability baseline used during manual adjudication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imaging.adjudication_audit import (
    AUTOMATIC_CROP_LEDGER,
    AUTOMATIC_CROP_SUMMARY,
    AdjudicationAuditError,
    audit_automatic_crops,
)

pytestmark = pytest.mark.public_portable


def _crops(tmp_path: Path, *, acquisitions: int = 2) -> Path:
    output_root = tmp_path / "v3_frozen_full"
    for index in range(1, acquisitions + 1):
        bundle = output_root / "crops" / f"acquisition_{index:04d}"
        bundle.mkdir(parents=True, exist_ok=True)
        for position in ("screen_left", "screen_right"):
            np.save(
                bundle / f"{position}_uint16.npy",
                np.full((8, 8), index * 10, dtype=np.uint16),
                allow_pickle=False,
            )
        (bundle / "record.json").write_text(
            json.dumps({"acquisition_index": index}), encoding="utf-8"
        )
    return output_root


def test_the_first_pass_creates_the_baseline_ledger(tmp_path: Path) -> None:
    output_root = _crops(tmp_path)

    summary = audit_automatic_crops(output_root=output_root, workers=2)

    assert summary["action"] == "ledger_created"
    assert summary["all_unchanged"] is True
    assert summary["panel_crops"] == 4
    assert summary["completion_records"] == 2
    assert summary["files"] == 6
    assert (output_root / AUTOMATIC_CROP_LEDGER).is_file()
    assert (output_root / AUTOMATIC_CROP_SUMMARY).is_file()


def test_an_unchanged_dataset_verifies_against_its_baseline(tmp_path: Path) -> None:
    output_root = _crops(tmp_path)
    audit_automatic_crops(output_root=output_root, workers=2)

    summary = audit_automatic_crops(output_root=output_root, workers=2)

    assert summary["action"] == "ledger_verified"
    assert summary["all_unchanged"] is True
    assert (summary["added"], summary["removed"], summary["modified"]) == (0, 0, 0)


def test_a_modified_crop_is_detected(tmp_path: Path) -> None:
    output_root = _crops(tmp_path)
    audit_automatic_crops(output_root=output_root, workers=2)
    target = output_root / "crops/acquisition_0001/screen_left_uint16.npy"
    np.save(target, np.zeros((8, 8), dtype=np.uint16), allow_pickle=False)

    summary = audit_automatic_crops(output_root=output_root, workers=2)

    assert summary["all_unchanged"] is False
    assert summary["modified"] == 1
    assert "crops/acquisition_0001/screen_left_uint16.npy" in summary["changed_examples"]


def test_a_removed_or_added_artifact_is_detected(tmp_path: Path) -> None:
    output_root = _crops(tmp_path)
    audit_automatic_crops(output_root=output_root, workers=2)
    (output_root / "crops/acquisition_0002/screen_right_uint16.npy").unlink()
    np.save(
        output_root / "crops/acquisition_0002/extra_uint16.npy",
        np.ones((8, 8), dtype=np.uint16),
        allow_pickle=False,
    )

    summary = audit_automatic_crops(output_root=output_root, workers=2)

    assert summary["removed"] == 1
    assert summary["added"] == 1
    assert summary["all_unchanged"] is False


def test_strict_mode_refuses_to_continue_after_a_change(tmp_path: Path) -> None:
    output_root = _crops(tmp_path)
    audit_automatic_crops(output_root=output_root, workers=2)
    np.save(
        output_root / "crops/acquisition_0001/screen_right_uint16.npy",
        np.zeros((8, 8), dtype=np.uint16),
        allow_pickle=False,
    )

    with pytest.raises(AdjudicationAuditError, match="changed since the baseline"):
        audit_automatic_crops(output_root=output_root, workers=2, strict=True)

    # The summary is still published so an operator can see what moved.
    summary = json.loads((output_root / AUTOMATIC_CROP_SUMMARY).read_text(encoding="utf-8"))
    assert summary["modified"] == 1


def test_a_corrected_crop_in_the_overrides_tree_does_not_disturb_the_baseline(
    tmp_path: Path,
) -> None:
    output_root = _crops(tmp_path)
    audit_automatic_crops(output_root=output_root, workers=2)
    bundle = output_root / "overrides/acquisition_0001"
    bundle.mkdir(parents=True, exist_ok=True)
    np.save(
        bundle / "screen_left_uint16.npy",
        np.full((8, 8), 999, dtype=np.uint16),
        allow_pickle=False,
    )

    summary = audit_automatic_crops(output_root=output_root, workers=2)

    assert summary["all_unchanged"] is True
    assert summary["files"] == 6


def test_a_missing_crop_tree_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(AdjudicationAuditError, match="does not exist"):
        audit_automatic_crops(output_root=tmp_path / "absent")
