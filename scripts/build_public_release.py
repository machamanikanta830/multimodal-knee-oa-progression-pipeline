#!/usr/bin/env python3
"""Fail-closed, non-destructive public release staging utility.

Reads the authoritative machine-readable manifest at `release/public_release_manifest.json`,
verifies all allowlist and denylist constraints, ensures strict destination and source
symlink safety, and stages an exact-membership public release candidate.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path

SENTINEL_FILENAME = ".builder_sentinel"
EXPECTED_SENTINEL_CONTENT = (
    "BUILDER_STAGING_SENTINEL_V1\n"
    "project: multimodal-knee-oa-progression-pipeline\n"
    "builder: scripts/build_public_release.py\n"
)
UNSUITABLE_DESTINATIONS = {
    Path("/"),
    Path("/tmp"),
    Path("/var"),
    Path("/private/tmp"),
    Path("/private/var"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/etc"),
    Path("/System"),
    Path("/Applications"),
    Path("/Library"),
}


class ReleaseBuilderError(Exception):
    """Base exception for release builder errors."""


class DestinationSafetyError(ReleaseBuilderError):
    """Raised when destination path violates safety constraints."""


class SymlinkSafetyError(ReleaseBuilderError):
    """Raised when an unsafe symlink is encountered."""


class ManifestSecurityError(ReleaseBuilderError):
    """Raised when manifest contains invalid or forbidden entries."""


class ExactMembershipError(ReleaseBuilderError):
    """Raised when staged tree does not exactly match allowlist."""


def validate_sentinel_file(sentinel_path: Path) -> None:
    """Validate that sentinel exists, is not a symlink, and contains exact expected content."""
    if sentinel_path.is_symlink():
        raise SymlinkSafetyError(f"Builder sentinel at {sentinel_path} is an unsafe symbolic link!")
    if not sentinel_path.is_file():
        raise DestinationSafetyError(
            f"Destination lacks regular {SENTINEL_FILENAME} file. "
            "Refusing to clean or overwrite non-empty directory."
        )
    try:
        content = sentinel_path.read_text(encoding="utf-8")
    except Exception as e:
        raise DestinationSafetyError(f"Failed to read sentinel at {sentinel_path}: {e}") from e

    if content != EXPECTED_SENTINEL_CONTENT:
        raise DestinationSafetyError(
            f"Sentinel content mismatch in {sentinel_path}. "
            "Refusing to clean directory not verified as builder-owned."
        )


def validate_allowlist_path_syntax(rel_path: str) -> None:
    """Validate path syntax strictly: reject absolute paths, traversal (.. or .), backslashes."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ManifestSecurityError(
            f"Malformed allowlist entry (empty or non-string): {rel_path!r}"
        )

    # Reject absolute paths
    if (
        rel_path.startswith("/")
        or rel_path.startswith("\\")
        or Path(rel_path).is_absolute()
        or Path(rel_path).drive
    ):
        raise ManifestSecurityError(f"Absolute paths forbidden in allowlist: {rel_path}")

    # Reject backslashes
    if "\\" in rel_path:
        raise ManifestSecurityError(f"Backslashes forbidden in allowlist path: {rel_path}")

    # Reject traversal tokens and empty segments
    segments = rel_path.split("/")
    for seg in segments:
        if seg in (".", ".."):
            raise ManifestSecurityError(
                f"Traversal component {seg!r} forbidden in allowlist path: {rel_path}"
            )
        if not seg:
            raise ManifestSecurityError(
                f"Empty path segment (double slash) forbidden in allowlist path: {rel_path}"
            )

    # Ensure normalized POSIX path matches original exactly
    norm = os.path.normpath(rel_path)
    if norm != rel_path or norm.startswith("..") or norm.startswith("/"):
        raise ManifestSecurityError(
            f"Allowlist path must be in canonical normalized POSIX form: {rel_path} (normalized: {norm})"
        )


def load_manifest(manifest_path: Path) -> dict:

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing release manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "allowlist" not in data or "denylist_patterns" not in data:
        raise ValueError(f"Invalid manifest structure in {manifest_path}")
    return data


def is_denied(rel_path: str, denylist_patterns: list[str]) -> bool:
    """Case-insensitive pattern matching against denylist."""
    posix_path = rel_path.replace("\\", "/").lower()
    path_obj = Path(posix_path)
    file_name = path_obj.name

    for pattern in denylist_patterns:
        norm_pattern = pattern.replace("\\", "/").lower()

        # Check direct match or filename match
        if fnmatch.fnmatch(posix_path, norm_pattern) or fnmatch.fnmatch(file_name, norm_pattern):
            return True

        # Check directory wildcard matches
        if norm_pattern.endswith("/**") and posix_path.startswith(norm_pattern[:-3]):
            return True
        if norm_pattern.endswith("/*") and posix_path.startswith(norm_pattern[:-2]):
            return True

        # Extension check if pattern is *.ext
        if norm_pattern.startswith("*.") and file_name.endswith(norm_pattern[1:]):
            return True

    return False


def validate_destination_safety(repo_root: Path, target_dir: Path, clean: bool = False) -> None:
    """Rigorous preflight destination safety checks."""
    try:
        home_dir = Path.home().resolve()
    except Exception:
        home_dir = None

    resolved_repo = repo_root.resolve()

    # Reject repo root itself
    if target_dir == repo_root or target_dir == resolved_repo:
        raise DestinationSafetyError(
            f"Target directory {target_dir} cannot be identical to repository root!"
        )

    # Resolve target directory if it exists or its deepest existing parent
    curr = target_dir
    while not curr.exists() and curr.parent != curr:
        curr = curr.parent

    if target_dir.is_symlink():
        raise SymlinkSafetyError(f"Target directory {target_dir} cannot itself be a symbolic link!")

    # Check for symlink components in destination ancestry (allowing standard OS tmp aliases)
    system_allowed_symlinks = {Path("/tmp"), Path("/var/tmp")}
    probe = target_dir
    while probe != probe.parent:
        if probe.is_symlink() and probe not in system_allowed_symlinks:
            raise SymlinkSafetyError(
                f"Destination path component {probe} is an unsafe symbolic link!"
            )
        probe = probe.parent

    resolved_target = target_dir.resolve()

    # Reject filesystem root
    if resolved_target == Path(resolved_target.anchor) or str(resolved_target) in ("/", ""):
        raise DestinationSafetyError("Target directory cannot be filesystem root!")

    # Reject user home directory
    if home_dir and (resolved_target == home_dir or target_dir == Path.home()):
        raise DestinationSafetyError("Target directory cannot be user home directory!")

    # Reject broad shared / system paths
    for unsuitable in UNSUITABLE_DESTINATIONS:
        if resolved_target == unsuitable.resolve() or target_dir == unsuitable:
            raise DestinationSafetyError(
                f"Target directory cannot be top-level shared path: {target_dir}"
            )

    # Reject repository descendant (inside repo)
    if resolved_target.is_relative_to(resolved_repo):
        raise DestinationSafetyError(
            f"Target directory {target_dir} cannot be inside repository {repo_root}!"
        )

    # Reject repository ancestor
    if resolved_repo.is_relative_to(resolved_target):
        raise DestinationSafetyError(
            f"Target directory {target_dir} cannot be an ancestor of repository {repo_root}!"
        )

    # If target directory exists, verify cleaning safety
    if target_dir.exists():
        existing_entries = list(target_dir.iterdir())
        if existing_entries:
            sentinel_path = target_dir / SENTINEL_FILENAME
            validate_sentinel_file(sentinel_path)
            if not clean:
                raise DestinationSafetyError(
                    f"Destination {target_dir} exists and contains previous release. "
                    "Specify --clean to wipe and restage."
                )


def validate_source_symlink_safety(repo_root: Path, rel_path: str) -> Path:
    """Assert source file and all path ancestors are regular files, not symlinks."""
    resolved_repo = repo_root.resolve()
    src_file = repo_root / rel_path

    # Check file itself
    if src_file.is_symlink():
        raise SymlinkSafetyError(f"Allowlisted source {rel_path} is a symbolic link!")

    # Check all ancestor components between repo_root and src_file
    curr = src_file.parent
    while curr != repo_root and curr != resolved_repo and curr.is_relative_to(repo_root):
        if curr.is_symlink():
            raise SymlinkSafetyError(
                f"Ancestor directory {curr} of source file {rel_path} is a symbolic link!"
            )
        curr = curr.parent

    resolved_src = src_file.resolve()
    if not resolved_src.is_relative_to(resolved_repo):
        raise SymlinkSafetyError(
            f"Source file {rel_path} resolves outside repository root: {resolved_src}"
        )

    return src_file


def build_release(
    repo_root: Path, target_dir: Path, manifest_path: Path, clean: bool = False
) -> int:
    """Stage public release candidate enforcing complete preflight before any mutation."""
    repo_root = repo_root.resolve()

    # Step 1: Load manifest
    print(f"Loading release manifest from {manifest_path}...")
    manifest = load_manifest(manifest_path)

    # Step 2: Validate manifest schema
    if "allowlist" not in manifest or "denylist_patterns" not in manifest:
        raise ManifestSecurityError(f"Manifest missing required keys in {manifest_path}")

    allowlist = manifest["allowlist"]
    denylist = manifest["denylist_patterns"]

    if not isinstance(allowlist, list) or not isinstance(denylist, list):
        raise ManifestSecurityError("Manifest allowlist and denylist_patterns must be lists")

    # Step 3: Validate allowlist uniqueness
    seen = set()
    duplicates = []
    for item in allowlist:
        if item in seen:
            duplicates.append(item)
        seen.add(item)

    if duplicates:
        raise ManifestSecurityError(f"Duplicate entries found in allowlist: {duplicates}")

    print(f"Authoritative allowlist contains {len(allowlist)} unique files.")

    # Step 4: Validate denylist conflicts
    for item in allowlist:
        if not isinstance(item, str):
            raise ManifestSecurityError(f"Allowlist item must be a string: {item!r}")
        if is_denied(item, denylist):
            raise ManifestSecurityError(
                f"Security violation: allowlist contains denied patterns: {item}"
            )

    # Steps 5-8: Validate every allowlist path syntactically
    for item in allowlist:
        validate_allowlist_path_syntax(item)

    # Steps 9-13: Validate all source paths
    for item in allowlist:
        src_file = validate_source_symlink_safety(repo_root, item)
        if not src_file.is_file():
            raise FileNotFoundError(f"Allowlisted file missing from repository: {item}")

    # Step 14: Validate all destination paths hypothetically
    resolved_target = target_dir.resolve()
    for item in allowlist:
        dst = target_dir / item
        if dst.is_symlink():
            raise SymlinkSafetyError(f"Destination target {dst} is an unsafe symbolic link!")
        abs_dst = Path(os.path.abspath(resolved_target / item))
        if not abs_dst.is_relative_to(resolved_target) or abs_dst == resolved_target:
            raise DestinationSafetyError(f"Destination path escape attempt: {item}")

    # Steps 15-16: Validate destination safety and sentinel
    validate_destination_safety(repo_root, target_dir, clean=clean)

    print("Preflight validation complete: 100% of allowlist entries verified.")

    # Step 17: ONLY AFTER ALL PREFLIGHT PASSES may any deletion or copy occur!
    if target_dir.exists() and clean:
        print(f"Cleaning approved previous staging directory {target_dir}...")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Write builder sentinel with deterministic format
    sentinel = target_dir / SENTINEL_FILENAME
    sentinel.write_text(EXPECTED_SENTINEL_CONTENT, encoding="utf-8")

    copied_count = 0
    for rel_path in allowlist:
        src = repo_root / rel_path
        dst = target_dir / rel_path

        # Verify dst is strictly inside target_dir
        if not dst.resolve().is_relative_to(target_dir.resolve()):
            raise DestinationSafetyError(
                f"Path escape attempt: {rel_path} resolves outside {target_dir}"
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_count += 1

    print(f"Successfully copied {copied_count} allowlisted files to {target_dir}.")

    # Post-staging EXACT MEMBERSHIP AUDIT (H04/H05/H06)
    actual_files = set()
    for root, _, files in os.walk(target_dir):
        for fname in files:
            p = Path(root) / fname
            rel = str(p.relative_to(target_dir)).replace("\\", "/")
            actual_files.add(rel)

    expected_files = set(allowlist) | {SENTINEL_FILENAME}

    extra_files = actual_files - expected_files
    missing_staged = expected_files - actual_files

    if extra_files:
        raise ExactMembershipError(
            f"Exact membership failure! Unexpected files present in staging tree: {sorted(extra_files)}"
        )
    if missing_staged:
        raise ExactMembershipError(
            f"Exact membership failure! Expected allowlisted files missing from staged tree: {sorted(missing_staged)}"
        )

    # Post-copy denylist audit as defense-in-depth
    post_violations = [f for f in actual_files if is_denied(f, denylist)]
    if post_violations:
        raise ManifestSecurityError(
            f"Post-copy denylist violation detected in destination: {post_violations}"
        )

    print(
        f"Exact membership verification PASSED: exactly {len(actual_files)} files staged "
        f"({len(allowlist)} allowlist + 1 sentinel)."
    )
    return copied_count


def main():
    parser = argparse.ArgumentParser(description="Build clean public release candidate.")
    parser.add_argument("destination", type=Path, help="Target directory for release candidate.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/public_release_manifest.json"),
        help="Path to authoritative manifest.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean target directory before copying if it contains a verified builder sentinel.",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    manifest = repo_root / args.manifest

    try:
        count = build_release(repo_root, args.destination, manifest, clean=args.clean)
        print(f"PUBLIC RELEASE STAGING COMPLETE: {count} files staged in {args.destination}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Release staging aborted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
