"""Synthetic tests for frozen full-cohort preprocessing orchestration."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

import imaging.full_preprocessing as full
from imaging.preprocessing import CropGeometry, JointLocalization

pytestmark = pytest.mark.public_portable


def _inputs(tmp_path: Path) -> dict[str, Path]:
    references = ["selected/a.tar.gz", "selected/b.tar.gz"]
    urls = ["s3://synthetic/selected/a.tar.gz", "s3://synthetic/selected/b.tar.gz"]
    reference_path = tmp_path / "references.txt"
    url_path = tmp_path / "urls.txt"
    reference_path.write_text("\n".join(references) + "\n", encoding="utf-8")
    url_path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    archive_root = tmp_path / "archives"
    for number, reference in enumerate(references, start=1):
        path = archive_root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{number}".encode())
    metadata_path = tmp_path / "package.csv.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["PACKAGE_FILE_ID", "NDA_S3_URL", "FILE_SIZE", "DOWNLOAD_ALIAS"],
        )
        writer.writeheader()
        for number, (reference, url) in enumerate(zip(references, urls, strict=True), start=1):
            writer.writerow(
                {
                    "PACKAGE_FILE_ID": number,
                    "NDA_S3_URL": url,
                    "FILE_SIZE": (archive_root / reference).stat().st_size,
                    "DOWNLOAD_ALIAS": reference,
                }
            )
    return {
        "references": reference_path,
        "urls": url_path,
        "metadata": metadata_path,
        "archives": archive_root,
        "ledger": tmp_path / "source_archives.parquet",
    }


def test_full_download_audit_requires_exact_sizes_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(full, "EXPECTED_ACQUISITIONS", 2)

    result = full.audit_full_download(
        references_path=inputs["references"],
        urls_path=inputs["urls"],
        package_metadata_path=inputs["metadata"],
        archive_root=inputs["archives"],
        ledger_path=inputs["ledger"],
    )

    ledger = pd.read_parquet(inputs["ledger"])
    assert result["download_gate_passed"]
    assert result["package_size_matches"] == 2
    assert ledger["archive_sha256"].nunique() == 2


def test_full_download_audit_stops_for_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(full, "EXPECTED_ACQUISITIONS", 2)
    final = inputs["archives"] / "selected/b.tar.gz"
    final.rename(final.with_name(final.name + ".partial"))

    with pytest.raises(full.FullPreprocessingError, match="Full download gate failed"):
        full.audit_full_download(
            references_path=inputs["references"],
            urls_path=inputs["urls"],
            package_metadata_path=inputs["metadata"],
            archive_root=inputs["archives"],
            ledger_path=inputs["ledger"],
        )


def test_padding_outside_validation_envelope_is_queued() -> None:
    location = JointLocalization(row=100, column=100, score=1.0, robust_score=2.0)
    geometry = CropGeometry(
        requested_rows=1067,
        requested_columns=1067,
        padding_top=0,
        padding_bottom=0,
        padding_left=234,
        padding_right=0,
    )

    record = full._location_record(location, geometry, "screen_left")

    assert record["horizontal_padding_mm"] == pytest.approx(35.1)
    assert record["excessive_padding"] is True


def test_visual_qc_review_flags_failures_without_changing_crops(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    qc_root = output_root / "qc"
    manifest_root = output_root / "manifests"
    qc_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    selection = pd.DataFrame(
        {
            "qc_number": [1, 2, 3],
            "acquisition_index": [10, 20, 30],
            "manufacturer": ["A", "A", "B"],
            "scanner_model": ["M1", "M1", "M2"],
            "spacing_mm": [0.15, 0.15, 0.2],
            "dimensions": ["100x200", "100x200", "120x240"],
            "image_release": ["R1", "R1", "R2"],
            "qc_category": ["Y", "Y", "NR"],
            "laterality_state": ["AMBIGUOUS", "AMBIGUOUS", "CONFLICTING"],
            "maximum_padding_mm": [0.0, 30.0, 2.0],
        }
    )
    selection.to_parquet(qc_root / "qc_selection_manifest.parquet", index=False)
    panels = pd.DataFrame(
        {
            "acquisition_index": [10, 10, 20, 20, 30, 30],
            "excessive_padding": [False, False, True, False, False, False],
        }
    )
    panel_path = tmp_path / "standardized.parquet"
    panels.to_parquet(panel_path, index=False)
    review = pd.DataFrame(
        {
            "qc_number": [1, 2, 3],
            "localization_visual_state": ["SUCCESS", "FAILED", "BORDERLINE"],
            "crop_visual_state": ["ADEQUATE", "INADEQUATE", "BORDERLINE"],
            "reason_code": ["OK", "MISLOCALIZED", "QUESTIONABLE"],
        }
    )
    review_path = tmp_path / "review.csv"
    review.to_csv(review_path, index=False)
    (manifest_root / "final_summary.json").write_text(json.dumps({"storage": {}}), encoding="utf-8")

    result = full.apply_visual_qc_review(
        review_path=review_path,
        output_root=output_root,
        standardized_manifest_path=panel_path,
    )

    updated = pd.read_parquet(panel_path)
    failures = pd.read_csv(output_root / "exceptions/localization_failure.csv")
    summary = json.loads((manifest_root / "final_summary.json").read_text(encoding="utf-8"))
    assert result["localization"]["success_or_adequate"] == 1
    assert result["localization"]["borderline"] == 1
    assert result["localization"]["failed_or_inadequate"] == 1
    assert (
        updated.loc[updated["acquisition_index"].eq(20), "visual_localization_state"]
        .eq("FAILED")
        .all()
    )
    assert failures["acquisition_index"].tolist() == [20]
    assert summary["full_set_status"] == "STOPPED_FOR_HUMAN_REVIEW"
    assert summary["ready_for_analysis_cohort_linkage"] is False
