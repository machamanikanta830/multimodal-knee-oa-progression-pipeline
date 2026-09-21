"""Synthetic tests for safe, privacy-preserving tar extraction."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from imaging.archive_extract import (
    UnsafeArchiveError,
    safe_extract_archive,
    summarize_archives,
)

pytestmark = pytest.mark.public_portable


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo("synthetic")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def _write_root_marker_archive(path: Path) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("./")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        info = tarfile.TarInfo("./image")
        info.size = 5
        archive.addfile(info, io.BytesIO(b"DICOM"))
    return path


def test_safe_archive_extracts_to_new_separate_destination(tmp_path: Path) -> None:
    source = _write_archive(tmp_path / "source.tar.gz", [("synthetic/image.dcm", b"DICOM")])
    destination = tmp_path / "extracted"

    contents = safe_extract_archive(source, destination)

    assert source.is_file()
    assert source.read_bytes()
    assert (destination / "synthetic" / "image.dcm").read_bytes() == b"DICOM"
    assert contents.members == 2
    assert contents.regular_files == 1
    assert contents.directories == 1
    summary = summarize_archives([contents])
    assert summary["archives_extracted"] == 1
    assert summary["privacy"]["archive_names_serialized"] is False


@pytest.mark.parametrize("unsafe_name", ["../escape.dcm", "/absolute.dcm"])
def test_rejects_path_traversal_without_partial_output(tmp_path: Path, unsafe_name: str) -> None:
    source = _write_archive(tmp_path / "private-source.tar.gz", [(unsafe_name, b"unsafe")])
    destination = tmp_path / "must-not-exist"

    with pytest.raises(UnsafeArchiveError) as error:
        safe_extract_archive(source, destination)

    assert not destination.exists()
    assert "private-source" not in str(error.value)


def test_rejects_links_and_does_not_serialize_member_names(tmp_path: Path) -> None:
    source = tmp_path / "links.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        link = tarfile.TarInfo("PRIVATE-PARTICIPANT-LINK")
        link.type = tarfile.SYMTYPE
        link.linkname = "target"
        archive.addfile(link)

    with pytest.raises(UnsafeArchiveError) as error:
        safe_extract_archive(source, tmp_path / "output")

    assert "PRIVATE-PARTICIPANT-LINK" not in str(error.value)
    assert "PRIVATE-PARTICIPANT-LINK" not in json.dumps(summarize_archives([], failures=1))


def test_refuses_existing_destination(tmp_path: Path) -> None:
    source = _write_archive(tmp_path / "source.tar.gz", [("synthetic/image", b"value")])
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        safe_extract_archive(source, destination)


def test_accepts_benign_current_directory_root_marker(tmp_path: Path) -> None:
    source = _write_root_marker_archive(tmp_path / "source.tar.gz")
    destination = tmp_path / "extracted"

    contents = safe_extract_archive(source, destination)

    assert (destination / "image").read_bytes() == b"DICOM"
    assert contents.members == 2
    assert contents.regular_files == 1
