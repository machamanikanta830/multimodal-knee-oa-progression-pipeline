"""Tests for the read-only inventory utility using temporary synthetic files only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.inventory import inventory_directory, main

pytestmark = pytest.mark.public_portable


def _records_by_path(directory: Path):
    return {record.path: record for record in inventory_directory(directory)}


def test_recursively_reports_csv_structure_and_unsupported_file(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    csv_path = nested / "synthetic.csv"
    original = b"participant_id,pain_score\n101,7\n102,8\n"
    csv_path.write_bytes(original)
    (tmp_path / "image.bin").write_bytes(b"synthetic-binary")

    records = _records_by_path(tmp_path)

    table = records["nested/synthetic.csv"]
    assert table.extension == ".csv"
    assert table.size_bytes == len(original)
    assert table.status == "tabular"
    assert table.row_count == 2
    assert table.column_count == 2
    assert table.column_names == ("participant_id", "pain_score")
    assert records["image.bin"].status == "unsupported"
    assert csv_path.read_bytes() == original


def test_tsv_reader_streams_quoted_multiline_record(tmp_path: Path) -> None:
    (tmp_path / "synthetic.tsv").write_text(
        'participant_id\tnote\n101\t"line one\nline two"\n',
        encoding="utf-8",
    )

    table = _records_by_path(tmp_path)["synthetic.tsv"]

    assert table.row_count == 1
    assert table.column_count == 2
    assert table.column_names == ("participant_id", "note")


def test_empty_csv_has_zero_rows_and_columns(tmp_path: Path) -> None:
    (tmp_path / "empty.csv").write_bytes(b"")

    table = _records_by_path(tmp_path)["empty.csv"]

    assert table.status == "tabular"
    assert table.row_count == 0
    assert table.column_count == 0
    assert table.column_names == ()


def test_xlsx_reports_active_sheet_structure_without_values(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Synthetic"
    worksheet.append(["participant_id", "score"])
    worksheet.append([101, "DO_NOT_PRINT_XLSX_VALUE"])
    workbook.save(tmp_path / "synthetic.xlsx")
    workbook.close()

    table = _records_by_path(tmp_path)["synthetic.xlsx"]

    assert table.status == "tabular"
    assert table.row_count == 1
    assert table.column_names == ("participant_id", "score")
    assert "DO_NOT_PRINT_XLSX_VALUE" not in json.dumps(table.as_dict())


def test_parquet_uses_metadata_without_exposing_values(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    table_path = tmp_path / "synthetic.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "participant_id": [101, 102],
                "comment": ["DO_NOT_PRINT_PARQUET_VALUE", "also synthetic"],
            }
        ),
        table_path,
    )

    record = _records_by_path(tmp_path)["synthetic.parquet"]

    assert record.status == "tabular"
    assert record.row_count == 2
    assert record.column_names == ("participant_id", "comment")
    assert "DO_NOT_PRINT_PARQUET_VALUE" not in json.dumps(record.as_dict())


def test_json_cli_never_prints_raw_row_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "synthetic.csv").write_text(
        "participant_id,comment\n101,DO_NOT_PRINT_RAW_VALUE\n",
        encoding="utf-8",
    )

    exit_code = main([str(tmp_path), "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["files"][0]["column_names"] == ["participant_id", "comment"]
    assert "DO_NOT_PRINT_RAW_VALUE" not in output


def test_non_directory_input_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.csv"
    file_path.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        inventory_directory(file_path)
