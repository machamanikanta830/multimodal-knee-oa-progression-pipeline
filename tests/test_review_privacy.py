"""Tests for redaction of burned-in detector margins on the review rendering path."""

from __future__ import annotations

import numpy as np
import pytest

from imaging.review_privacy import (
    BURNED_IN_MARGIN_MM,
    crop_rows_inside_margin,
    redact_horizontal_margins,
)

pytestmark = pytest.mark.public_portable


def test_margins_are_masked_in_millimetres_not_pixels() -> None:
    coarse = np.full((100, 40), 700, dtype=np.uint16)
    fine = np.full((400, 40), 700, dtype=np.uint16)

    coarse_redacted, coarse_record = redact_horizontal_margins(
        coarse, row_spacing_mm=1.0, fill_value=0, margin_mm=10.0
    )
    fine_redacted, fine_record = redact_horizontal_margins(
        fine, row_spacing_mm=0.25, fill_value=0, margin_mm=10.0
    )

    assert coarse_record.top_rows == 10
    assert fine_record.top_rows == 40
    assert not coarse_redacted[:10].any()
    assert not coarse_redacted[-10:].any()
    assert (coarse_redacted[10:-10] == 700).all()
    assert not fine_redacted[:40].any()
    assert (fine_redacted[40:-40] == 700).all()


def test_burned_in_text_near_an_edge_is_removed() -> None:
    panel = np.full((240, 60), 500, dtype=np.uint16)
    panel[228:233, 10:50] = 4000  # printed label 3.5 mm from the bottom edge at 0.5 mm/pixel
    panel[120, :] = 3000  # central anatomy that must survive

    redacted, record = redact_horizontal_margins(
        panel, row_spacing_mm=0.5, fill_value=500, margin_mm=BURNED_IN_MARGIN_MM
    )

    assert record.top_rows == 70
    assert redacted.max() == 3000
    assert (redacted[120] == 3000).all()
    assert not (redacted == 4000).any()


def test_redaction_never_overlaps_itself_on_a_short_image() -> None:
    image = np.full((9, 5), 3, dtype=np.uint16)

    redacted, record = redact_horizontal_margins(
        image, row_spacing_mm=1.0, fill_value=0, margin_mm=100.0
    )

    assert record.top_rows == 4
    assert record.bottom_rows == 4
    assert record.rows_redacted == 8
    assert redacted[4].tolist() == [3, 3, 3, 3, 3]


def test_source_array_is_not_modified() -> None:
    image = np.full((80, 10), 250, dtype=np.uint16)

    redacted, _ = redact_horizontal_margins(image, row_spacing_mm=1.0, fill_value=0, margin_mm=5.0)

    assert (image == 250).all()
    assert not redacted[:5].any()


def test_record_reports_the_redacted_fraction() -> None:
    image = np.zeros((200, 10), dtype=np.uint16)

    _, record = redact_horizontal_margins(image, row_spacing_mm=1.0, fill_value=0, margin_mm=20.0)

    assert record.rows_redacted == 40
    assert record.fraction_redacted == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("row_spacing_mm", "margin_mm"),
    [(0.0, 10.0), (-1.0, 10.0), (float("nan"), 10.0), (1.0, -5.0), (1.0, float("inf"))],
)
def test_invalid_geometry_is_rejected(row_spacing_mm: float, margin_mm: float) -> None:
    with pytest.raises(ValueError):
        redact_horizontal_margins(
            np.zeros((10, 10), dtype=np.uint16),
            row_spacing_mm=row_spacing_mm,
            fill_value=0,
            margin_mm=margin_mm,
        )


def test_non_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        redact_horizontal_margins(
            np.zeros((4, 4, 3), dtype=np.uint16), row_spacing_mm=1.0, fill_value=0
        )


def test_centred_crop_reports_no_margin_rows() -> None:
    assert (
        crop_rows_inside_margin(
            center_row=1000,
            crop_rows=200,
            panel_rows=2000,
            row_spacing_mm=1.0,
            margin_mm=35.0,
        )
        == 0
    )


def test_crop_reaching_the_bottom_edge_reports_margin_rows() -> None:
    # Crop spans rows 1900..2100 of a 2000-row panel, so 65 rows fall in the bottom 35 mm band.
    assert (
        crop_rows_inside_margin(
            center_row=2000,
            crop_rows=200,
            panel_rows=2000,
            row_spacing_mm=1.0,
            margin_mm=35.0,
        )
        == 35
    )


def test_crop_reaching_the_top_edge_reports_margin_rows() -> None:
    assert (
        crop_rows_inside_margin(
            center_row=10,
            crop_rows=100,
            panel_rows=2000,
            row_spacing_mm=1.0,
            margin_mm=35.0,
        )
        == 35
    )
