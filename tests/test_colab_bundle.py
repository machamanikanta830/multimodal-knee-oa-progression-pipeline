"""Synthetic bundle tests only; never train or predict using real imaging data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imaging.artifact_io import atomic_json, sha256_file
from modeling import colab_bundle as bundle
from modeling.colab_assets import runtime
from multimodal.freeze import DatasetIntegrityError, publish_json, publish_parquet

pytestmark = pytest.mark.public_portable

TOY = {"TRAIN": (4, 2, 2), "VALIDATION": (4, 2, 2)}


def recatalog(root):
    path = root / "bundle_freeze.json"
    freeze = json.loads(path.read_text())
    freeze["artifacts_sha256"] = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "bundle_freeze.json"
    }
    atomic_json(freeze, path)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    lineage = bundle.stage_code(root)
    rows, people = [], []
    for role, prefix in (("TRAIN", "T"), ("VALIDATION", "V")):
        for person in (0, 1):
            pid = f"{prefix}{person}"
            people.append(
                {
                    "participant_id": pid,
                    "split": role,
                    "eligible_knee_count": 2,
                    "primary_event_knee_count": 2 * person,
                }
            )
            for knee in ("1", "2"):
                relative = f"{pid}/{knee}.npy"
                path = root / "images" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.arange(16, dtype=np.uint16).reshape(4, 4))
                rows.append(
                    {
                        "participant_id": pid,
                        "knee_side_code": knee,
                        "baseline_visit": "V00",
                        "composite_progression": bool(person),
                        "effective_crop_relative_path": relative,
                        "effective_crop_sha256": sha256_file(path),
                        "effective_provenance": "AUTO_PASS",
                        "anatomical_side": "R" if knee == "1" else "L",
                        "split": role,
                    }
                )
    manifest = bundle.normalize(pd.DataFrame(rows)[runtime.MANIFEST_COLUMNS])
    for name, frame in (
        ("image_manifest", manifest),
        ("participant_split_projection", bundle.normalize(pd.DataFrame(people))),
        (
            "knee_split_projection",
            manifest[[*runtime.tabular.IDENTITY, "anatomical_side", "split"]],
        ),
    ):
        publish_parquet(frame, root / f"metadata/{name}.parquet")
    publish_json(runtime.SOURCES, root / "source_expectations.json")
    weights = root / "pretrained/densenet121-a639ec97.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"synthetic payload; never loaded as a model")
    monkeypatch.setattr(runtime, "WEIGHT_SHA", sha256_file(weights))
    publish_json(
        {
            "version": runtime.VERSION,
            "source_hashes": runtime.SOURCES,
            "approved_code_hashes": runtime.CODE,
            "verbatim_metrics": lineage,
            "artifacts_sha256": {},
        },
        root / "bundle_freeze.json",
    )
    recatalog(root)
    return root


def test_only_development_memberships_no_TEST_identifiers(toy):
    result = runtime.verify(toy, TOY)
    assert result["TRAIN"] == result["VALIDATION"] == 4 and result["TEST"] == 0
    manifest, people, knees = runtime.tables(toy, TOY)
    assert len(manifest) == len(knees) == 8 and len(people) == 4
    assert set(manifest.split) == {"TRAIN", "VALIDATION"}


@pytest.mark.parametrize(
    "fault",
    [
        "TEST",
        "duplicate_person",
        "missing_knee",
        "wrong_side",
        "wrong_participant",
        "wrong_label",
        "wrong_visit",
        "unresolved",
        "duplicate_image",
    ],
)
def test_malformed_metadata_fails_even_with_recomputed_catalog(toy, fault):
    path = toy / "metadata/image_manifest.parquet"
    frame = pd.read_parquet(path)
    if fault == "TEST":
        frame.loc[0, "split"] = "TEST"
    elif fault == "duplicate_person":
        p = toy / "metadata/participant_split_projection.parquet"
        people = pd.read_parquet(p)
        pd.concat([people, people.iloc[:1]], ignore_index=True).to_parquet(p, index=False)
    elif fault == "missing_knee":
        frame = frame.iloc[1:].reset_index(drop=True)
    elif fault == "wrong_side":
        frame.loc[0, "anatomical_side"] = "L"
    elif fault == "wrong_participant":
        frame.loc[0, "participant_id"] = "LOCKED_TEST_ID"
    elif fault == "wrong_label":
        frame.loc[0, "composite_progression"] = True
    elif fault == "wrong_visit":
        frame.loc[0, "baseline_visit"] = "V06"
    elif fault == "unresolved":
        frame.loc[0, "effective_provenance"] = "UNRESOLVED"
    else:
        frame.loc[1, "effective_crop_relative_path"] = frame.loc[0, "effective_crop_relative_path"]
    frame.to_parquet(path, index=False)
    recatalog(toy)
    with pytest.raises((DatasetIntegrityError, AssertionError)):
        runtime.verify(toy, TOY)


@pytest.mark.parametrize(
    "fault",
    [
        "extra_image",
        "missing_image",
        "mutated_image",
        "absolute_path",
        "symlink",
        "changed_config",
        "changed_expectation",
    ],
)
def test_artifact_and_image_tampering_fails_closed(toy, fault, tmp_path):
    image = toy / "images/T0/1.npy"
    if fault == "extra_image":
        (toy / "images/TEST_IMAGE.npy").write_bytes(b"not permitted")
        recatalog(toy)
    elif fault == "missing_image":
        image.unlink()
    elif fault == "mutated_image":
        image.write_bytes(b"mutated")
        recatalog(toy)
    elif fault == "absolute_path":
        p = toy / "metadata/image_manifest.parquet"
        frame = pd.read_parquet(p)
        frame.loc[0, "effective_crop_relative_path"] = str(image)
        frame.to_parquet(p, index=False)
        recatalog(toy)
    elif fault == "symlink":
        external = tmp_path / "external.npy"
        external.write_bytes(image.read_bytes())
        image.unlink()
        image.symlink_to(external)
    elif fault == "changed_config":
        (toy / "configs/image_model_v1.yaml").write_text("{}")
        recatalog(toy)
    else:
        p = toy / "source_expectations.json"
        v = json.loads(p.read_text())
        v["dataset"] = "0" * 64
        atomic_json(v, p)
        recatalog(toy)
    with pytest.raises(DatasetIntegrityError):
        runtime.verify(toy, TOY)


def test_relative_paths_resolve_after_relocation(toy, tmp_path):
    relocated = tmp_path / "different_location"
    toy.rename(relocated)
    runtime.verify(relocated, TOY)
    manifest, _, _ = runtime.tables(relocated, TOY)
    for relative in manifest.effective_crop_relative_path:
        assert runtime.safe_path(relocated, "images/" + relative).is_file()


def test_source_copy_exact_and_existing_mismatch_refused(tmp_path):
    source, dest = tmp_path / "source.npy", tmp_path / "copy.npy"
    source.write_bytes(b"exact synthetic bytes")
    sha = sha256_file(source)
    bundle.copy_exact(source, dest, sha)
    bundle.copy_exact(source, dest, sha)
    assert sha256_file(source) == sha256_file(dest) == sha
    dest.write_bytes(b"changed")
    with pytest.raises(DatasetIntegrityError, match="overwrite"):
        bundle.copy_exact(source, dest, sha)
    with pytest.raises(DatasetIntegrityError, match="Source SHA"):
        bundle.copy_exact(source, tmp_path / "bad.npy", "0" * 64)


def test_scientific_source_and_helpers_packaged_verbatim(toy):
    assert sha256_file(toy / "src/modeling/image.py") == runtime.CODE["src/modeling/image.py"]
    assert runtime.blocks(toy / "src/modeling/tabular.py", bundle.HELPERS) == runtime.blocks(
        Path(bundle.tabular.__file__), bundle.HELPERS
    )
    assert not (toy / "src/cohort").exists()
    assert "CatBoostClassifier" not in (toy / "src/modeling/tabular.py").read_text()


def test_archive_member_hashes_and_idempotent_reuse(toy, tmp_path, monkeypatch):
    original = runtime.verify
    monkeypatch.setattr(runtime, "verify", lambda root: original(root, TOY))
    path = tmp_path / "bundle.tar.gz"
    first = bundle.archive(toy, path)
    second = bundle.archive(toy, path)
    assert first == second and first["archive_member_payload_SHA_verification_passed"]
    assert first["archive_sha256"] == sha256_file(path)
    assert first["TRAIN_images"] == first["VALIDATION_images"] == 4


def test_installer_pins_versions_and_refuses_local_production():
    shell = (bundle.ASSETS / "colab_setup.sh").read_text()
    assert "set -euo pipefail" in shell and "Do not substitute versions" in shell
    code = (bundle.ASSETS / "run_training.py").read_text()
    assert 'choices=("cuda",)' in code and "--plan" in code
    assert "runtime.environment(ROOT, cuda=True)" in code


@pytest.mark.parametrize("fault", ["version", "no_cuda", "wrong_gpu"])
def test_target_environment_stops_before_training(toy, monkeypatch, fault):
    import torch

    publish_json({"torch": "2.14.0", "torchvision": "0.29.0"}, toy / "dependency_versions.json")
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: (
            "0.0" if fault == "version" else {"torch": "2.14.0", "torchvision": "0.29.0"}[name]
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: fault == "wrong_gpu")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NOT_THE_APPROVED_GPU")
    monkeypatch.setattr(
        torch,
        "ones",
        lambda *a, **kw: pytest.fail("CUDA allocation after failed target verification"),
    )
    with pytest.raises(DatasetIntegrityError, match="STOP"):
        runtime.environment(toy, cuda=True)


@pytest.mark.parametrize("name", ["NVIDIA A100-SXM4-40GB", "NVIDIA L4", "Tesla T4"])
def test_approved_cuda_hardware_accepted_and_recorded(toy, monkeypatch, name):
    import torch

    publish_json({"torch": "2.14.0", "torchvision": "0.29.0"}, toy / "dependency_versions.json")
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda key: {"torch": "2.14.0", "torchvision": "0.29.0"}[key],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: name)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    report = runtime.environment(toy, cuda=True)
    assert report["GPU"] == name and selected == [0]
    from modeling import image

    assert image.device_info(torch.device("cuda"))["CUDA_name"] == name


def test_a100_preferred_without_changing_training(toy, monkeypatch):
    import torch

    publish_json({"torch": "2.14.0", "torchvision": "0.29.0"}, toy / "dependency_versions.json")
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda key: {"torch": "2.14.0", "torchvision": "0.29.0"}[key],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    names = ["Tesla T4", "NVIDIA L4", "NVIDIA A100 80GB PCIe"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: names[index])
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    report = runtime.environment(toy, cuda=True)
    assert selected == [2] and report["GPU"] == names[2]


@pytest.mark.parametrize(
    "name", ["NVIDIA V100", "NVIDIA RTX A1000", "CPU", "TPU", "FAKE T4", "NVIDIA L40"]
)
def test_nonapproved_hardware_name_rejected(name):
    assert runtime.approved_gpu_priority(name) is None
