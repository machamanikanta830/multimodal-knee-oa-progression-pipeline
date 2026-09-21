"""Explicit laterality-evidence states for bilateral radiographs.

The module never derives anatomical side from screen position alone. A mapping is returned only
when acquisition-level evidence is sufficient and non-conflicting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LateralityState(str, Enum):
    """Review state for anatomical side assignment."""

    CONFIDENT = "CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"


PanelMapping = tuple[str, str]
VALID_MAPPING: PanelMapping = ("R", "L")
REVERSED_MAPPING: PanelMapping = ("L", "R")


@dataclass(frozen=True, slots=True)
class LateralityAssessment:
    """Privacy-safe evidence result for one bilateral acquisition."""

    state: LateralityState
    screen_left_anatomical_side: str | None
    screen_right_anatomical_side: str | None
    reason: str


def _validate_mapping(mapping: PanelMapping | None, label: str) -> None:
    if mapping is not None and mapping not in {VALID_MAPPING, REVERSED_MAPPING}:
        raise ValueError(f"{label} must map screen panels to one R and one L knee")


def assess_laterality(
    *,
    bilateral_acquisition: bool,
    dicom_laterality: str | None,
    image_laterality: str | None,
    marker_mapping: PanelMapping | None,
    orientation_mapping: PanelMapping | None = None,
    documented_convention_mapping: PanelMapping | None = None,
) -> LateralityAssessment:
    """Classify evidence and return a mapping only for confident cases.

    `Laterality` and `ImageLaterality` values of R/L conflict with a confirmed bilateral image;
    B is compatible with bilateral structure but does not map the two screen panels.
    """

    _validate_mapping(marker_mapping, "Marker mapping")
    _validate_mapping(orientation_mapping, "Orientation mapping")
    _validate_mapping(documented_convention_mapping, "Convention mapping")
    if not bilateral_acquisition:
        return LateralityAssessment(
            LateralityState.AMBIGUOUS,
            None,
            None,
            "acquisition_not_confirmed_bilateral",
        )

    technical_values = {
        str(value).strip().upper()
        for value in (dicom_laterality, image_laterality)
        if value not in (None, "")
    }
    if technical_values.intersection({"R", "L"}):
        return LateralityAssessment(
            LateralityState.CONFLICTING,
            None,
            None,
            "unilateral_dicom_tag_on_bilateral_image",
        )
    mappings = [
        mapping
        for mapping in (marker_mapping, orientation_mapping, documented_convention_mapping)
        if mapping is not None
    ]
    if len(set(mappings)) > 1:
        return LateralityAssessment(
            LateralityState.CONFLICTING,
            None,
            None,
            "screen_mapping_evidence_disagrees",
        )
    if not mappings:
        return LateralityAssessment(
            LateralityState.AMBIGUOUS,
            None,
            None,
            "no_screen_to_anatomical_mapping_evidence",
        )
    screen_left, screen_right = mappings[0]
    return LateralityAssessment(
        LateralityState.CONFIDENT,
        screen_left,
        screen_right,
        "consistent_bilateral_mapping_evidence",
    )
