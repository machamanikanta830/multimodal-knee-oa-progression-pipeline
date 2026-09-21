"""Redaction of burned-in detector margins for rendered manual-review images.

Several OAI baseline acquisitions carry machine-printed text burned into the pixel data near the
top or bottom detector edge. The strings observed include label fragments and identifier-like
sequences, so any rendered review image must not show them, even though the frozen preprocessing
policy itself reads only geometry and never reads that text.

Measured on a stratified sample of every scanner family in the baseline set, stroke-scale
high-frequency energy above the family background is confined to the top and bottom edges and
reaches at most 33 mm inward. No family shows comparable structure at the left or right edge, so
only horizontal bands are redacted and the full width of the acquisition stays visible.

This module is used only on the rendering path. Stored uint16 crops keep their original pixels: the
frozen crop specification is unchanged, and redaction is a presentation-layer guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Deepest observed burned-in structure was 33 mm from an edge; 35 mm keeps a small safety margin.
BURNED_IN_MARGIN_MM = 35.0


@dataclass(frozen=True, slots=True)
class MarginRedaction:
    """Record of which rows a rendered review image had masked, in pixels and millimetres."""

    margin_mm: float
    top_rows: int
    bottom_rows: int
    rows_total: int

    @property
    def rows_redacted(self) -> int:
        return self.top_rows + self.bottom_rows

    @property
    def fraction_redacted(self) -> float:
        return self.rows_redacted / self.rows_total if self.rows_total else 0.0


def redact_horizontal_margins(
    image: np.ndarray,
    *,
    row_spacing_mm: float,
    fill_value: float,
    margin_mm: float = BURNED_IN_MARGIN_MM,
) -> tuple[np.ndarray, MarginRedaction]:
    """Return a copy of ``image`` with its top and bottom margin bands replaced by ``fill_value``.

    ``row_spacing_mm`` is the physical row spacing of ``image``, so the same millimetre policy
    applies identically to a native-resolution acquisition and to a resampled panel. The bands never
    overlap: for an image shorter than twice the margin the two halves split the available rows.
    """

    if image.ndim != 2 or image.size == 0:
        raise ValueError("Margin redaction requires a non-empty two-dimensional image")
    if not np.isfinite(row_spacing_mm) or row_spacing_mm <= 0:
        raise ValueError("Row spacing must be finite and positive")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("Margin must be finite and non-negative")

    rows = image.shape[0]
    band = min(int(round(margin_mm / row_spacing_mm)), rows // 2)
    redacted = np.array(image, copy=True)
    if band > 0:
        redacted[:band] = fill_value
        redacted[rows - band :] = fill_value
    return redacted, MarginRedaction(
        margin_mm=float(margin_mm),
        top_rows=band,
        bottom_rows=band,
        rows_total=int(rows),
    )


def crop_rows_inside_margin(
    *,
    center_row: int,
    crop_rows: int,
    panel_rows: int,
    row_spacing_mm: float,
    margin_mm: float = BURNED_IN_MARGIN_MM,
) -> int:
    """Count crop rows that originate inside a redacted panel margin.

    The stored crop is never redacted, so this is reported rather than enforced: it quantifies how
    many crops carry detector-margin pixels that could contain burned-in text.
    """

    band = min(int(round(margin_mm / row_spacing_mm)), panel_rows // 2)
    if band <= 0:
        return 0
    top = center_row - crop_rows // 2
    bottom = top + crop_rows
    overlap_top = max(0, min(bottom, band) - max(top, 0))
    overlap_bottom = max(0, min(bottom, panel_rows) - max(top, panel_rows - band))
    return int(overlap_top + overlap_bottom)
