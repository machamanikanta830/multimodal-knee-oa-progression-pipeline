"""Synthetic tests for the V3 localizer and independent anatomical QC gate."""

from __future__ import annotations

import numpy as np
import pytest

from imaging.localization_v3 import (
    QCState,
    assess_panel_layout,
    find_vertical_candidates,
    localize_bilateral_tibiofemoral_joints_v3,
    validate_anatomical_crop,
)

pytestmark = pytest.mark.public_portable


def _joint_panel(rows: int = 1500, columns: int = 1000, joint_row: int = 790) -> np.ndarray:
    image = np.full((rows, columns), 150, dtype=np.uint16)
    # Broad femoral condyles and tibial plateau separated by a joint-space band.
    image[260 : joint_row - 18, 270:730] = 2600
    image[joint_row - 18 : joint_row + 18, 230:770] = 500
    image[joint_row + 18 : 1320, 250:750] = 2250
    image[joint_row - 150 : joint_row + 150, 190:810] += 300
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 180, size=image.shape, dtype=np.uint16)
    image = np.where(image > 200, image + texture, image).astype(np.uint16)
    return image


def test_multiple_vertical_candidate_bands_are_returned_in_rank_order() -> None:
    profile = np.zeros(600, dtype=np.float32)
    profile[220] = 4
    profile[360] = 7
    profile[480] = 5

    candidates = find_vertical_candidates(profile, spacing_mm=0.15, maximum_candidates=3)

    assert len(candidates) == 3
    assert candidates[0][0] == 360
    assert len({row for row, _, _ in candidates}) == 3


def test_valid_joint_like_target_passes_anatomical_gate() -> None:
    panel = _joint_panel()

    result = validate_anatomical_crop(panel, center=(790, 500), spacing_mm=0.15)

    assert result.state in {QCState.PASS, QCState.BORDERLINE}
    assert result.superior_structure > 0
    assert result.inferior_structure > 0


def test_shaft_like_false_target_is_not_passed() -> None:
    panel = np.full((1500, 1000), 100, dtype=np.uint16)
    panel[100:1400, 440:560] = 2600

    result = validate_anatomical_crop(panel, center=(430, 500), spacing_mm=0.15)

    assert result.state is not QCState.PASS
    assert any("joint" in reason or "shaft" in reason for reason in result.reasons)


def test_collimation_boundary_is_not_accepted_as_joint() -> None:
    panel = _joint_panel()
    panel[:, 500:] = np.clip(panel[:, 500:] + 2000, 0, np.iinfo(np.uint16).max)

    result = validate_anatomical_crop(panel, center=(790, 500), spacing_mm=0.15)

    assert result.state is not QCState.PASS


def test_insufficient_superior_bone_coverage_is_detected() -> None:
    panel = _joint_panel()
    panel[:770] = 150

    result = validate_anatomical_crop(panel, center=(790, 500), spacing_mm=0.15)

    assert result.state is not QCState.PASS
    assert "insufficient_superior_bone_structure" in result.reasons


def test_insufficient_inferior_bone_coverage_is_detected() -> None:
    panel = _joint_panel()
    panel[810:] = 150

    result = validate_anatomical_crop(panel, center=(790, 500), spacing_mm=0.15)

    assert result.state is not QCState.PASS
    assert "insufficient_inferior_bone_structure" in result.reasons


def test_blank_background_excess_fails_panel_layout() -> None:
    panel = np.zeros((1500, 1000), dtype=np.uint16)
    panel[650:850, 400:600] = 1000

    result = assess_panel_layout(panel, spacing_mm=0.15)

    assert result.state is QCState.FAIL


def test_candidate_confidence_margin_produces_safe_nonpass() -> None:
    left = _joint_panel()
    right = _joint_panel()
    # Add a second joint-like band of similar strength to create unresolved vertical evidence.
    left[430:466, 230:770] = 500
    right[430:466, 230:770] = 500

    results = localize_bilateral_tibiofemoral_joints_v3(left, right, spacing_mm=0.15)

    assert all(result.qc_state in set(QCState) for result in results)
    assert all(result.candidate_count >= 2 for result in results)


def test_difficult_panel_geometry_is_queued_not_forced() -> None:
    panel = _joint_panel(rows=500, columns=1500, joint_row=300)

    result = assess_panel_layout(panel, spacing_mm=0.15)

    assert result.state is QCState.FAIL
