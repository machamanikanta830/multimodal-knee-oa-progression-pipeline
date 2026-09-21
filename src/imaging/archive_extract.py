"""Safe extraction helpers for access-controlled OAI pilot archives.

The functions reject links, special files, absolute paths, parent traversal, and duplicate
member targets. Callers can therefore unpack each source archive into a separate ignored local
directory without ever modifying the archive.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class UnsafeArchiveError(ValueError):
    """Raised before extraction when a tar archive has an unsafe structure."""


@dataclass(frozen=True, slots=True)
class ArchiveContents:
    """Aggregate-only archive content information."""

    members: int
    regular_files: int
    directories: int
    other_members: int
    declared_file_bytes: int
    member_depths: tuple[int, ...]


def _validated_members(archive: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], ArchiveContents]:
    members = archive.getmembers()
    targets: set[PurePosixPath] = set()
    depths: list[int] = []
    for member in members:
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise UnsafeArchiveError("Archive contains an unsafe member path")
        if not (member.isfile() or member.isdir()):
            raise UnsafeArchiveError("Archive contains a link or special-file member")
        normalized = PurePosixPath(*(part for part in relative.parts if part not in ("", ".")))
        if not normalized.parts:
            if member.isdir() and member.name in {".", "./"}:
                depths.append(0)
                continue
            raise UnsafeArchiveError("Archive contains an empty member target")
        if normalized in targets:
            raise UnsafeArchiveError("Archive contains a duplicate member target")
        targets.add(normalized)
        depths.append(max(len(normalized.parts) - 1, 0))
    contents = ArchiveContents(
        members=len(members),
        regular_files=sum(member.isfile() for member in members),
        directories=sum(member.isdir() for member in members),
        other_members=sum(not (member.isfile() or member.isdir()) for member in members),
        declared_file_bytes=sum(member.size for member in members if member.isfile()),
        member_depths=tuple(depths),
    )
    return members, contents


def inspect_archive(path: str | Path) -> ArchiveContents:
    """Inspect an archive without extracting it or returning member names."""

    try:
        with tarfile.open(path, mode="r:gz") as archive:
            _, contents = _validated_members(archive)
            return contents
    except (OSError, tarfile.TarError) as error:
        raise UnsafeArchiveError("Archive could not be read safely") from error


def safe_extract_archive(path: str | Path, destination: str | Path) -> ArchiveContents:
    """Extract one validated archive to a new destination atomically.

    Existing destinations are refused so this operation cannot overwrite prior extraction output.
    Member names are never returned or included in errors.
    """

    target = Path(destination)
    if target.exists():
        raise FileExistsError("Extraction destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        try:
            with tarfile.open(path, mode="r:gz") as archive:
                members, contents = _validated_members(archive)
                for member in members:
                    relative = PurePosixPath(member.name)
                    parts = [part for part in relative.parts if part not in ("", ".")]
                    if not parts:
                        continue
                    normalized = Path(*parts)
                    output = temporary / normalized
                    if member.isdir():
                        output.mkdir(parents=True, exist_ok=True)
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise UnsafeArchiveError("A regular archive member could not be read")
                    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with source, os.fdopen(descriptor, "wb") as handle:
                        shutil.copyfileobj(source, handle)
        except (OSError, tarfile.TarError) as error:
            raise UnsafeArchiveError("Archive could not be extracted safely") from error
        temporary.replace(target)
        return contents
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def summarize_archives(contents: Iterable[ArchiveContents], *, failures: int = 0) -> dict[str, Any]:
    """Return aggregate archive-content distributions with no archive or member names."""

    records = list(contents)

    def counts(values: Iterable[int]) -> dict[str, int]:
        return {str(key): value for key, value in sorted(Counter(values).items())}

    return {
        "archives_considered": len(records) + failures,
        "archives_extracted": len(records),
        "extraction_failures": failures,
        "members_per_archive": counts(record.members for record in records),
        "regular_files_per_archive": counts(record.regular_files for record in records),
        "directories_per_archive": counts(record.directories for record in records),
        "other_members": sum(record.other_members for record in records),
        "declared_extracted_file_bytes": sum(record.declared_file_bytes for record in records),
        "member_depths": counts(depth for record in records for depth in record.member_depths),
        "privacy": {
            "archive_names_serialized": False,
            "member_names_serialized": False,
        },
    }


def contents_as_dict(contents: ArchiveContents) -> dict[str, Any]:
    """Serialize aggregate content fields only."""

    return asdict(contents)
