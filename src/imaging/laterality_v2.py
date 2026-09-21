"""Conservative OAI bilateral-radiograph laterality evidence hierarchy.

The module does not read participant identifiers and does not treat a unilateral DICOM
Laterality/ImageLaterality value as authoritative for a confirmed bilateral acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from imaging.laterality import VALID_MAPPING, PanelMapping, _validate_mapping


class LateralityV2State(str, Enum):
    """Disposition for a proposed screen-panel to anatomical-side mapping."""

    CONFIDENT = "CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"


@dataclass(frozen=True, slots=True)
class LateralityV2Assessment:
    """Mapping decision with its strongest evidence and DICOM-tag interpretation."""

    state: LateralityV2State
    screen_left_anatomical_side: str | None
    screen_right_anatomical_side: str | None
    evidence_source: str
    reason: str
    unilateral_dicom_tag_artifact: bool


def _unilateral_tag_present(*values: str | None) -> bool:
    normalized = {
        str(value).strip().upper() for value in values if value is not None and str(value).strip()
    }
    return bool(normalized.intersection({"R", "L"}))


def assess_laterality_v2(
    *,
    bilateral_acquisition: bool,
    midpoint_separation_clean: bool,
    burned_in_marker_mapping: PanelMapping | None = None,
    anatomical_mapping: PanelMapping | None = None,
    validated_screen_mapping: PanelMapping | None = None,
    screen_mapping_validation_approved: bool = False,
    exception_detector_passed: bool = False,
    dicom_laterality: str | None = None,
    image_laterality: str | None = None,
) -> LateralityV2Assessment:
    """Apply the V2 hierarchy without forcing an unsupported anatomical assignment.

    Burned-in markers and anatomical evidence are primary image evidence. A screen-position
    convention can resolve an otherwise unmarked image only after an independent audit approves
    it and acquisition-level exception checks pass. Unilateral DICOM tags on confirmed bilateral
    images are recorded as a metadata artifact; they neither establish nor overturn a mapping.
    """

    _validate_mapping(burned_in_marker_mapping, "Burned-in marker mapping")
    _validate_mapping(anatomical_mapping, "Anatomical mapping")
    _validate_mapping(validated_screen_mapping, "Validated screen mapping")
    dicom_artifact = bilateral_acquisition and _unilateral_tag_present(
        dicom_laterality, image_laterality
    )
    if not bilateral_acquisition or not midpoint_separation_clean:
        return LateralityV2Assessment(
            LateralityV2State.AMBIGUOUS,
            None,
            None,
            "none",
            "bilateral_geometry_not_confirmed",
            dicom_artifact,
        )

    primary = [
        ("burned_in_marker", burned_in_marker_mapping),
        ("anatomical_fibular_orientation", anatomical_mapping),
    ]
    present_primary = [(source, mapping) for source, mapping in primary if mapping is not None]
    if len({mapping for _, mapping in present_primary}) > 1:
        return LateralityV2Assessment(
            LateralityV2State.CONFLICTING,
            None,
            None,
            "image_evidence",
            "primary_image_evidence_disagrees",
            dicom_artifact,
        )
    if present_primary:
        source, mapping = present_primary[0]
        if (
            validated_screen_mapping is not None
            and screen_mapping_validation_approved
            and mapping != validated_screen_mapping
        ):
            return LateralityV2Assessment(
                LateralityV2State.CONFLICTING,
                None,
                None,
                "image_vs_screen_rule",
                "primary_image_evidence_disagrees_with_validated_screen_rule",
                dicom_artifact,
            )
        assert mapping is not None
        return LateralityV2Assessment(
            LateralityV2State.CONFIDENT,
            mapping[0],
            mapping[1],
            source,
            "primary_image_evidence",
            dicom_artifact,
        )

    if (
        validated_screen_mapping is not None
        and screen_mapping_validation_approved
        and exception_detector_passed
    ):
        return LateralityV2Assessment(
            LateralityV2State.CONFIDENT,
            validated_screen_mapping[0],
            validated_screen_mapping[1],
            "validated_oai_screen_position",
            "audited_screen_rule_with_exception_checks",
            dicom_artifact,
        )

    return LateralityV2Assessment(
        LateralityV2State.AMBIGUOUS,
        None,
        None,
        "none",
        "insufficient_mapping_evidence",
        dicom_artifact,
    )


def assess_validated_oai_screen_rule(
    *,
    bilateral_acquisition: bool,
    midpoint_separation_clean: bool,
    exception_detector_passed: bool,
    dicom_laterality: str | None = None,
    image_laterality: str | None = None,
) -> LateralityV2Assessment:
    """Apply the independently audited OAI mapping while preserving exception behavior."""

    return assess_laterality_v2(
        bilateral_acquisition=bilateral_acquisition,
        midpoint_separation_clean=midpoint_separation_clean,
        validated_screen_mapping=VALID_MAPPING,
        screen_mapping_validation_approved=True,
        exception_detector_passed=exception_detector_passed,
        dicom_laterality=dicom_laterality,
        image_laterality=image_laterality,
    )
