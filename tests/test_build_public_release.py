"""Comprehensive synthetic tests for hardened release builder safeguards.

Verifies destination safety, symlink rejection, case-insensitive denylist checks,
preflight assertions, and exact-membership enforcement in isolated temporary directories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_public_release import (  # noqa: E402
    EXPECTED_SENTINEL_CONTENT,
    SENTINEL_FILENAME,
    DestinationSafetyError,
    ExactMembershipError,
    ManifestSecurityError,
    SymlinkSafetyError,
    build_release,
    is_denied,
    validate_destination_safety,
    validate_source_symlink_safety,
)

pytestmark = pytest.mark.public_portable


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal synthetic repo with allowlisted files and manifest."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Create synthetic source files
    (repo / "README.md").write_text("# Synthetic Readme\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")

    # Create manifest
    manifest_data = {
        "manifest_version": "test_v1",
        "allowlist": ["README.md", "src/main.py"],
        "denylist_patterns": ["*.parquet", "*.dcm", "*.pt"],
    }
    manifest_path = repo / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    dest = tmp_path / "dest"
    return repo, manifest_path, dest


def test_successful_build_exact_membership(synthetic_repo):
    repo, manifest_path, dest = synthetic_repo
    count = build_release(repo, dest, manifest_path)
    assert count == 2
    assert (dest / "README.md").is_file()
    assert (dest / "src" / "main.py").is_file()
    assert (dest / SENTINEL_FILENAME).is_file()

    # Check re-cleaning works when sentinel is present
    count2 = build_release(repo, dest, manifest_path, clean=True)
    assert count2 == 2


def test_reject_ancestor_destination(synthetic_repo):
    repo, manifest_path, _ = synthetic_repo
    ancestor = repo.parent
    with pytest.raises(DestinationSafetyError, match="ancestor"):
        validate_destination_safety(repo, ancestor)


def test_reject_repo_root_destination(synthetic_repo):
    repo, _, _ = synthetic_repo
    with pytest.raises(DestinationSafetyError, match="identical"):
        validate_destination_safety(repo, repo)


def test_reject_descendant_destination(synthetic_repo):
    repo, _, _ = synthetic_repo
    inside = repo / "nested_dest"
    with pytest.raises(DestinationSafetyError, match="inside"):
        validate_destination_safety(repo, inside)


def test_reject_root_destination(synthetic_repo):
    repo, _, _ = synthetic_repo
    with pytest.raises(DestinationSafetyError, match="filesystem root"):
        validate_destination_safety(repo, Path("/"))


def test_reject_home_destination(synthetic_repo):
    repo, _, _ = synthetic_repo
    home = Path.home()
    with pytest.raises(DestinationSafetyError, match="home"):
        validate_destination_safety(repo, home)


def test_reject_nonempty_unapproved_destination(synthetic_repo):
    repo, manifest_path, dest = synthetic_repo
    dest.mkdir()
    (dest / "unrelated_file.txt").write_text("preexisting content\n", encoding="utf-8")

    # Without sentinel, must fail even with clean=True
    with pytest.raises(DestinationSafetyError, match="lacks regular .builder_sentinel"):
        build_release(repo, dest, manifest_path, clean=True)


def test_reject_source_symlink(synthetic_repo):
    repo, _, _ = synthetic_repo
    target = repo / "actual_file.txt"
    target.write_text("data\n", encoding="utf-8")

    link = repo / "symlink_file.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation not permitted on host")

    with pytest.raises(SymlinkSafetyError, match="symbolic link"):
        validate_source_symlink_safety(repo, "symlink_file.txt")


def test_reject_symlink_ancestor(synthetic_repo):
    repo, _, _ = synthetic_repo
    real_dir = repo / "real_dir"
    real_dir.mkdir()
    (real_dir / "target.py").write_text("code\n", encoding="utf-8")

    link_dir = repo / "link_dir"
    try:
        link_dir.symlink_to(real_dir)
    except OSError:
        pytest.skip("Symlink creation not permitted on host")

    with pytest.raises(SymlinkSafetyError, match="Ancestor directory"):
        validate_source_symlink_safety(repo, "link_dir/target.py")


def test_reject_source_symlink_escape(synthetic_repo, tmp_path):
    repo, _, _ = synthetic_repo
    external = tmp_path / "outside.txt"
    external.write_text("secret\n", encoding="utf-8")

    link = repo / "escape.txt"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("Symlink creation not permitted on host")

    with pytest.raises(SymlinkSafetyError):
        validate_source_symlink_safety(repo, "escape.txt")


def test_case_insensitive_denylist():
    patterns = ["*.dcm", "*.parquet", "*.pt"]
    assert is_denied("image.dcm", patterns) is True
    assert is_denied("image.DCM", patterns) is True
    assert is_denied("image.Dicom", ["*.dicom"]) is True
    assert is_denied("data/patient.PARQUET", patterns) is True
    assert is_denied("checkpoint.PT", patterns) is True
    assert is_denied("model.safetensors", patterns) is False


def test_reject_allowlist_denylist_conflict(synthetic_repo):
    repo, _, dest = synthetic_repo
    (repo / "data.PARQUET").write_text("forbidden\n", encoding="utf-8")

    manifest = {
        "allowlist": ["README.md", "data.PARQUET"],
        "denylist_patterns": ["*.parquet"],
    }
    mpath = repo / "conflict_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestSecurityError, match="denied patterns"):
        build_release(repo, dest, mpath)


def test_reject_duplicate_allowlist_entry(synthetic_repo):
    repo, _, dest = synthetic_repo
    manifest = {
        "allowlist": ["README.md", "README.md"],
        "denylist_patterns": ["*.parquet"],
    }
    mpath = repo / "dup_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestSecurityError, match="Duplicate"):
        build_release(repo, dest, mpath)


def test_reject_missing_allowlisted_source(synthetic_repo):
    repo, _, dest = synthetic_repo
    manifest = {
        "allowlist": ["README.md", "nonexistent.py"],
        "denylist_patterns": ["*.parquet"],
    }
    mpath = repo / "missing_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing from repository"):
        build_release(repo, dest, mpath)


def test_exact_membership_mismatch_detected(synthetic_repo, monkeypatch):
    repo, manifest_path, dest = synthetic_repo

    # Simulate an extra file appearing during staging
    orig_copy2 = __import__("shutil").copy2

    def rogue_copy2(src, dst):
        orig_copy2(src, dst)
        # Drop an extra file
        extra = dest / "rogue_extra.txt"
        extra.write_text("rogue", encoding="utf-8")

    monkeypatch.setattr("shutil.copy2", rogue_copy2)

    with pytest.raises(ExactMembershipError, match="Unexpected files present"):
        build_release(repo, dest, manifest_path)


def test_reject_absolute_allowlist_path_before_mutation(synthetic_repo):
    """Absolute paths must fail preflight without touching existing staging files."""
    repo, _, dest = synthetic_repo
    dest.mkdir()
    (dest / SENTINEL_FILENAME).write_text(EXPECTED_SENTINEL_CONTENT, encoding="utf-8")
    canary = dest / "important_staged_canary.txt"
    canary.write_text("must_not_be_deleted", encoding="utf-8")

    manifest = {
        "manifest_version": "test_v1",
        "allowlist": ["README.md", "/etc/hosts"],
        "denylist_patterns": ["*.parquet"],
    }
    mpath = repo / "abs_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestSecurityError, match="Absolute paths forbidden"):
        build_release(repo, dest, mpath, clean=True)

    # Staging directory and canary file must remain completely intact
    assert canary.is_file()
    assert canary.read_text(encoding="utf-8") == "must_not_be_deleted"


def test_reject_traversal_allowlist_path_before_mutation(synthetic_repo):
    """Traversal paths (..) must fail preflight without touching existing staging files."""
    repo, _, dest = synthetic_repo
    dest.mkdir()
    (dest / SENTINEL_FILENAME).write_text(EXPECTED_SENTINEL_CONTENT, encoding="utf-8")
    canary = dest / "important_staged_canary.txt"
    canary.write_text("must_not_be_deleted", encoding="utf-8")

    manifest = {
        "manifest_version": "test_v1",
        "allowlist": ["README.md", "src/../README.md"],
        "denylist_patterns": ["*.parquet"],
    }
    mpath = repo / "traversal_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestSecurityError, match="Traversal component"):
        build_release(repo, dest, mpath, clean=True)

    # Staging directory and canary file must remain completely intact
    assert canary.is_file()
    assert canary.read_text(encoding="utf-8") == "must_not_be_deleted"


def test_later_invalid_manifest_item_prevents_all_cleanup(synthetic_repo):
    """Any later-invalid manifest item must prevent all cleanup and copying."""
    repo, _, dest = synthetic_repo
    dest.mkdir()
    (dest / SENTINEL_FILENAME).write_text(EXPECTED_SENTINEL_CONTENT, encoding="utf-8")
    canary = dest / "important_staged_canary.txt"
    canary.write_text("must_not_be_deleted", encoding="utf-8")

    manifest = {
        "manifest_version": "test_v1",
        "allowlist": ["README.md", "src/main.py", "does_not_exist.py"],
        "denylist_patterns": ["*.parquet"],
    }
    mpath = repo / "later_invalid_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing from repository"):
        build_release(repo, dest, mpath, clean=True)

    # Staging directory and canary file must remain completely intact
    assert canary.is_file()
    assert canary.read_text(encoding="utf-8") == "must_not_be_deleted"
    assert not (dest / "src" / "main.py").exists()


def test_sentinel_valid_allows_clean(synthetic_repo):
    """Correct sentinel content permits controlled cleanup and restaging."""
    repo, manifest_path, dest = synthetic_repo
    dest.mkdir()
    (dest / SENTINEL_FILENAME).write_text(EXPECTED_SENTINEL_CONTENT, encoding="utf-8")
    (dest / "old_staged_file.txt").write_text("old", encoding="utf-8")

    count = build_release(repo, dest, manifest_path, clean=True)
    assert count == 2
    assert (dest / "README.md").is_file()
    assert not (dest / "old_staged_file.txt").exists()
    assert (dest / SENTINEL_FILENAME).read_text(encoding="utf-8") == EXPECTED_SENTINEL_CONTENT


def test_sentinel_invalid_content_rejects_clean(synthetic_repo):
    """Sentinel with unexpected content rejects cleanup and preserves directory."""
    repo, manifest_path, dest = synthetic_repo
    dest.mkdir()
    (dest / SENTINEL_FILENAME).write_text("arbitrary or malicious text", encoding="utf-8")
    canary = dest / "user_data.txt"
    canary.write_text("preserve_me", encoding="utf-8")

    with pytest.raises(DestinationSafetyError, match="Sentinel content mismatch"):
        build_release(repo, dest, manifest_path, clean=True)

    assert canary.is_file()
    assert canary.read_text(encoding="utf-8") == "preserve_me"


def test_sentinel_empty_rejects_clean(synthetic_repo):
    """Empty sentinel rejects cleanup and preserves directory."""
    repo, manifest_path, dest = synthetic_repo
    dest.mkdir()
    (dest / SENTINEL_FILENAME).write_text("", encoding="utf-8")
    canary = dest / "user_data.txt"
    canary.write_text("preserve_me", encoding="utf-8")

    with pytest.raises(DestinationSafetyError, match="Sentinel content mismatch"):
        build_release(repo, dest, manifest_path, clean=True)

    assert canary.is_file()
    assert canary.read_text(encoding="utf-8") == "preserve_me"


def test_sentinel_symlink_rejects_clean(synthetic_repo, tmp_path):
    """Symlink sentinel rejects cleanup and preserves directory."""
    repo, manifest_path, dest = synthetic_repo
    dest.mkdir()
    real_sentinel = tmp_path / "external_sentinel.txt"
    real_sentinel.write_text(EXPECTED_SENTINEL_CONTENT, encoding="utf-8")

    link_sentinel = dest / SENTINEL_FILENAME
    try:
        link_sentinel.symlink_to(real_sentinel)
    except OSError:
        pytest.skip("Symlink creation not supported on host")

    canary = dest / "user_data.txt"
    canary.write_text("preserve_me", encoding="utf-8")

    with pytest.raises(SymlinkSafetyError, match="unsafe symbolic link"):
        build_release(repo, dest, manifest_path, clean=True)

    assert canary.is_file()
    assert canary.read_text(encoding="utf-8") == "preserve_me"
