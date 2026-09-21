"""Shared synthetic fixtures for the manual-review test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

# The last synthetic case stands in for an acquisition whose laterality the frozen policy could not
# resolve, so the laterality-exception path is exercised alongside the ordinary queue.
EXCEPTION_CASE = 3


def build_review_queue(tmp_path: Path, *, cases: int = 3) -> Path:
    """Write a synthetic review queue with rendered previews and return its path."""

    output_root = tmp_path / "v3_frozen_full"
    rows = []
    for review_index in range(1, cases + 1):
        acquisition = 100 + review_index
        exception = review_index == EXCEPTION_CASE
        preview_root = output_root / "review/previews" / f"acquisition_{acquisition:04d}"
        preview_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "bilateral",
            "overlay_screen_left",
            "overlay_screen_right",
            "crop_screen_left",
            "crop_screen_right",
        ):
            Image.new("RGB", (40, 60), "black").save(preview_root / f"{name}.png")
        relative = preview_root.relative_to(output_root).as_posix()
        row = {
            "review_index": review_index,
            "acquisition_index": acquisition,
            "participant_id": f"NDAR_INV{review_index:08d}",
            "source_participant_id": f"90016{review_index:02d}",
            "accession_number": f"1660042{review_index:04d}",
            "associated_file_reference": f"image03/00m/0.C.2/secret_{review_index}.tar.gz",
            "automatic_state": "BORDERLINE",
            "review_reason_codes": "screen_left:limited_bone_coverage",
            "laterality_state": "AMBIGUOUS" if exception else "CONFIDENT",
            "laterality_reason": (
                "bilateral_geometry_not_confirmed"
                if exception
                else "audited_screen_rule_with_exception_checks"
            ),
            "structure_check_passed": not exception,
            "structure_check_reasons": (
                "limb_centers_are_too_close_for_a_bilateral_acquisition" if exception else ""
            ),
            "bilateral_preview_path": f"{relative}/bilateral.png",
            "queue_reason_count": 1,
        }
        for position in ("screen_left", "screen_right"):
            row.update(
                {
                    f"{position}_crop_path": f"crops/acquisition_{acquisition:04d}/"
                    f"{position}_uint16.npy",
                    f"{position}_overlay_path": f"{relative}/overlay_{position}.png",
                    f"{position}_crop_preview_path": f"{relative}/crop_{position}.png",
                    f"{position}_overlay_scale": 0.5,
                    f"{position}_overlay_header_pixels": 22,
                    f"{position}_resampled_rows": 1200,
                    f"{position}_resampled_columns": 800,
                    f"{position}_joint_row_resampled": 600,
                    f"{position}_joint_column_resampled": 420,
                    f"{position}_candidate_centers": json.dumps(
                        [{"row": 600, "column": 420, "score": 0.5}]
                    ),
                    f"{position}_qc_state": "BORDERLINE",
                    f"{position}_localization_state": "PASS",
                    f"{position}_layout_state": "PASS",
                    f"{position}_anatomical_validator_state": "BORDERLINE",
                    f"{position}_localization_confidence": 0.62,
                    f"{position}_horizontal_padding_mm": 0.0,
                    f"{position}_vertical_padding_mm": 1.5,
                    f"{position}_anatomical_side": (
                        None if exception else ("R" if position == "screen_left" else "L")
                    ),
                }
            )
        rows.append(row)
    queue_path = output_root / "review_queue.parquet"
    pd.DataFrame(rows).to_parquet(queue_path, index=False)
    return queue_path


@pytest.fixture
def review_queue(tmp_path: Path) -> Path:
    return build_review_queue(tmp_path)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags for pytest."""
    parser.addoption(
        "--run-historical-lifecycle",
        action="store_true",
        default=False,
        help="Run historical pre-TEST lifecycle tests that are superseded after final evaluation",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip historical pre-TEST lifecycle tests unless explicitly requested."""
    if not config.getoption("--run-historical-lifecycle"):
        skip_historical = pytest.mark.skip(
            reason="Historical lifecycle test: requires --run-historical-lifecycle flag to run (pre-TEST conditions superseded)"
        )
        for item in items:
            if "historical_lifecycle" in item.keywords:
                item.add_marker(skip_historical)
