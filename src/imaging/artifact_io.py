"""Atomic local-artifact writers and content hashing shared by versioned imaging runs.

Every writer replaces its target in one filesystem operation so an interrupted run cannot leave a
partially written manifest, summary, or ledger behind.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HASH_BLOCK_BYTES = 4 * 1024 * 1024


def json_default(value: Any) -> Any:
    """Serialize NumPy scalars that survive aggregation into summary dictionaries."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of a UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(value: Any, target: str | Path) -> Path:
    """Write sorted, indented JSON to ``target`` atomically."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".json", dir=destination.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_parquet(frame: pd.DataFrame, target: str | Path) -> Path:
    """Write a DataFrame to Parquet atomically."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".parquet", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_png(image: Any, target: str | Path) -> Path:
    """Write a Pillow-compatible image as PNG and publish it atomically."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        image.save(temporary, format="PNG", optimize=True)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class LogCorruptionError(ValueError):
    """Raised when a line-delimited log is damaged somewhere other than its final record."""


@dataclass(frozen=True, slots=True)
class LogTail:
    """A terminal fragment left behind when an append was interrupted mid-write."""

    valid_bytes: int
    fragment: bytes
    reason: str

    @property
    def fragment_bytes(self) -> int:
        return len(self.fragment)


def serialize_jsonl_record(value: Any) -> bytes:
    """Serialize one record completely before any file is touched.

    Preparing the whole payload up front keeps the time spent inside an append transaction to a
    single write, so a serialization error can never leave a partial line behind.
    """

    return (json.dumps(value, sort_keys=True, default=json_default) + "\n").encode("utf-8")


@contextmanager
def file_lock(target: str | Path) -> Iterator[int]:
    """Hold an exclusive advisory lock that serializes writers across processes.

    ``threading.Lock`` only orders threads inside one interpreter, so two review servers could
    otherwise read the same state and both believe they are recording a first decision. The lock
    lives in a sidecar file so it is independent of the log's own file descriptor lifetime.
    """

    lock_path = Path(f"{target}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def append_bytes(payload: bytes, target: str | Path) -> Path:
    """Append prepared bytes to a log durably, retrying a short write until it completes.

    Append-only logging keeps an interrupted reviewer session recoverable and makes an accidental
    whole-file overwrite impossible through the normal write path. Callers that need the read and
    the append to be one transaction must hold :func:`file_lock` around this call.
    """

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def append_jsonl(value: Any, target: str | Path) -> Path:
    """Serialize one record and append it durably."""

    return append_bytes(serialize_jsonl_record(value), target)


def read_jsonl(target: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a line-delimited JSON log, tolerating a missing file."""

    records, _ = read_jsonl_checked(target)
    yield from records


def read_jsonl_checked(target: str | Path) -> tuple[list[dict[str, Any]], LogTail | None]:
    """Read a line-delimited log, tolerating only an interrupted final record.

    A crash during an append can leave an unterminated or half-serialized record at end of file.
    That single terminal fragment is reported rather than raised, so prior decisions stay readable
    and a session can resume. Damage anywhere earlier in the log fails closed, because it cannot be
    explained by an interrupted append and may indicate edited or corrupted history.
    """

    source = Path(target)
    if not source.is_file():
        return [], None
    raw = source.read_bytes()
    if not raw:
        return [], None

    terminated, _, trailing = raw.rpartition(b"\n")
    if terminated == b"" and _ == b"":
        # No newline anywhere: the whole file is one unterminated fragment.
        complete_lines: list[bytes] = []
        tail_fragment = trailing
        valid_bytes = 0
    else:
        complete_lines = terminated.split(b"\n")
        tail_fragment = trailing
        valid_bytes = len(raw) - len(trailing)

    records: list[dict[str, Any]] = []
    for number, line in enumerate(complete_lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise LogCorruptionError(
                f"Log record {number} of {source.name} is malformed and is not a terminal "
                "interrupted append"
            ) from error

    tail: LogTail | None = None
    if tail_fragment.strip():
        # Unterminated trailing bytes mean the append never completed. Such a record is never
        # promoted to history even when it happens to parse, because a write that lost its newline
        # cannot be distinguished from one that stopped early, and it must be cleared before the
        # next append fuses onto it.
        tail = LogTail(
            valid_bytes=valid_bytes,
            fragment=tail_fragment,
            reason="interrupted_final_append",
        )
    return records, tail


def quarantine_log_tail(target: str | Path, tail: LogTail) -> Path:
    """Move an interrupted final fragment into a sidecar so appends can resume safely.

    The fragment is copied before the log is shortened, and only the fragment's own bytes are
    removed, so every previously valid record survives untouched. Leaving the fragment in place is
    not an option: the next append would fuse onto it and destroy a real record.
    """

    destination = Path(target)
    with destination.open("rb") as stream:
        stream.seek(tail.valid_bytes)
        observed = stream.read()
    if observed != tail.fragment:
        raise LogCorruptionError("The log changed while its interrupted tail was being recovered")
    sidecar = destination.with_suffix(f"{destination.suffix}.corrupt_tail")
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    entry = {
        "recovered_at_utc": stamp,
        "log": destination.name,
        "reason": tail.reason,
        "valid_bytes": tail.valid_bytes,
        "fragment_bytes": tail.fragment_bytes,
        "fragment_base64": base64.b64encode(tail.fragment).decode("ascii"),
    }
    append_bytes(serialize_jsonl_record(entry), sidecar)
    descriptor = os.open(destination, os.O_WRONLY)
    try:
        os.ftruncate(descriptor, tail.valid_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sidecar


def hash_paths(
    paths: Sequence[Path],
    *,
    relative_to: str | Path,
    workers: int = 4,
) -> pd.DataFrame:
    """Hash the given files into a deterministic ledger sorted by relative artifact path."""

    anchor = Path(relative_to)
    ordered = sorted(set(paths))

    def record(path: Path) -> dict[str, Any]:
        return {
            "artifact": path.relative_to(anchor).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    if not ordered:
        return pd.DataFrame(columns=["artifact", "bytes", "sha256"])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(record, ordered))
    return pd.DataFrame(records, columns=["artifact", "bytes", "sha256"])


def hash_tree(
    root: str | Path,
    *,
    relative_to: str | Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """Hash every regular file below ``root`` into a deterministic preservation ledger."""

    base = Path(root)
    anchor = Path(relative_to) if relative_to is not None else base
    files = [path for path in base.rglob("*") if path.is_file()]
    return hash_paths(files, relative_to=anchor, workers=workers)
