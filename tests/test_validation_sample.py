"""Synthetic tests for independent imaging-validation sample preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from imaging.validation_sample import (
    DIVERSITY_WEIGHTS,
    ValidationSelectionError,
    extract_image03_subset,
    select_validation_acquisitions,
)

pytestmark = pytest.mark.public_portable


def _candidates() -> pd.DataFrame:
    rows = []
    for grade in range(4):
        for knee_count in (1, 2):
            for replicate in range(20):
                row = {
                    "accession_number": f"A-{grade}-{knee_count}-{replicate:02d}",
                    "validation_kl_stratum": grade,
                    "eligible_knee_count": knee_count,
                }
                for offset, column in enumerate(DIVERSITY_WEIGHTS):
                    row[column] = f"{column}-{(replicate + grade + offset) % 5}"
                rows.append(row)
    return pd.DataFrame(rows)


def test_validation_selection_is_independent_balanced_and_deterministic() -> None:
    candidates = _candidates()
    development = {f"A-{grade}-{knee_count}-00" for grade in range(4) for knee_count in (1, 2)}

    first = select_validation_acquisitions(candidates, development)
    second = select_validation_acquisitions(candidates.sample(frac=1, random_state=91), development)

    assert len(first) == 128
    assert first["accession_number"].nunique() == 128
    assert not set(first["accession_number"]).intersection(development)
    assert first["accession_number"].tolist() == second["accession_number"].tolist()
    assert not first["selection_outcomes_read"].any()
    assert not first["development_pilot_overlap"].any()
    counts = first.groupby(["validation_kl_stratum", "eligible_knee_count"]).size()
    assert counts.eq(16).all()


def test_image03_subset_preserves_header_and_dictionary_bytes(tmp_path: Path) -> None:
    header = b"\xef\xbb\xbfsubjectkey\taccession_number\timage_file\r\n"
    dictionary = b"GUID\tAccession\tAssociated file\r\n"
    rows = {
        "A1": b"G1\tA1\tone.tar.gz\r\n",
        "A2": b"G2\tA2\ttwo.tar.gz\r\n",
        "A3": b"G3\tA3\tthree.tar.gz\r\n",
    }
    source = tmp_path / "image03.txt"
    source.write_bytes(header + dictionary + b"".join(rows.values()))

    chunks = extract_image03_subset(source, {"A1", "A3"})

    assert chunks[:2] == [header, dictionary]
    assert b"".join(chunks) == header + dictionary + rows["A1"] + rows["A3"]
    assert rows["A2"] not in chunks


def test_image03_subset_refuses_missing_or_duplicate_matches(tmp_path: Path) -> None:
    source = tmp_path / "image03.txt"
    source.write_bytes(
        b"subjectkey\taccession_number\timage_file\n"
        b"GUID\tAccession\tAssociated file\n"
        b"G1\tA1\tone.tar.gz\n"
        b"G1\tA1\tone-again.tar.gz\n"
    )

    with pytest.raises(ValidationSelectionError, match="more than once"):
        extract_image03_subset(source, {"A1"})
    with pytest.raises(ValidationSelectionError, match="missing 1"):
        extract_image03_subset(source, {"A2"})
