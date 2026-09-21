"""Synthetic-only image-model guards, preprocessing, fitting and checkpoint tests."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from imaging.artifact_io import sha256_file
from modeling import image, tabular
from multimodal.freeze import DatasetIntegrityError, load_config

pytestmark = pytest.mark.public_portable


@pytest.fixture
def config():
    return deepcopy(load_config(image.CONFIG))


@pytest.fixture
def data(tmp_path):
    parts = {}
    membership = {
        "T0": "TRAIN",
        "T1": "TRAIN",
        "V0": "VALIDATION",
        "V1": "VALIDATION",
        "LOCKED": "TEST",
    }
    for name, prefix in (("TRAIN", "T"), ("VALIDATION", "V")):
        rows = []
        for person in range(2):
            for knee in ("1", "2"):
                path = tmp_path / f"{prefix}{person}_{knee}.npy"
                ramp = np.arange(1067 * 1067, dtype=np.uint32).reshape(1067, 1067)
                array = (ramp % 60000).astype(np.uint16)
                if person:
                    array[350:650, 350:650] = 65535
                np.save(path, array)
                rows.append(
                    {
                        "participant_id": f"{prefix}{person}",
                        "knee_side_code": knee,
                        "baseline_visit": "V00",
                        tabular.TARGET: bool(person),
                        "effective_crop_relative_path": path.name,
                        "effective_crop_sha256": sha256_file(path),
                        "effective_provenance": "AUTO_PASS",
                    }
                )
        frame = pd.DataFrame(rows)
        for c in (
            *tabular.IDENTITY,
            "effective_crop_relative_path",
            "effective_crop_sha256",
            "effective_provenance",
        ):
            frame[c] = frame[c].astype("string")
        parts[name] = tabular.DevelopmentPartition(
            name,
            frame,
            membership,
            frozenset(frame[list(tabular.IDENTITY)].itertuples(index=False, name=None)),
        )
    dev = tabular.DevelopmentData(
        parts["TRAIN"],
        parts["VALIDATION"],
        {**tabular.APPROVED, "imaging_manifest": image.IMAGE_SHA},
    )
    result = image.ImageData(dev, tmp_path)
    result.validate()
    return result


def test_predeclared_candidate_and_augmentation_contract(config):
    image.validate_config(config)
    assert len(config["candidates"]) == 3
    for key, value in (
        ("input_size", 224),
        ("test_prediction_permitted", True),
        ("loss", "weighted"),
    ):
        c = deepcopy(config)
        c[key] = value
        with pytest.raises(DatasetIntegrityError):
            image.validate_config(c)
    config["augmentation"]["horizontal_flip"] = True
    with pytest.raises(DatasetIntegrityError):
        image.validate_config(config)


@pytest.mark.parametrize("name", ["dataset", "participants", "knees", "feature_groups"])
def test_all_approved_dataset_split_hashes_required(monkeypatch, name):
    names = {
        "final_multimodal_dataset.parquet": "dataset",
        "participant_split_manifest.parquet": "participants",
        "knee_split_manifest.parquet": "knees",
        "feature_groups.json": "feature_groups",
    }
    monkeypatch.setattr(image, "sha256_file", lambda path: image.IMAGE_SHA)
    monkeypatch.setattr(
        tabular,
        "sha256_file",
        lambda path: "0" * 64 if names[path.name] == name else tabular.APPROVED[names[path.name]],
    )
    with pytest.raises(DatasetIntegrityError, match="SHA mismatch"):
        image.load_image_data()


def test_approved_imaging_manifest_hash_required(monkeypatch):
    monkeypatch.setattr(image, "sha256_file", lambda path: "0" * 64)
    with pytest.raises(DatasetIntegrityError, match="manifest SHA mismatch"):
        image.load_image_data()


@pytest.mark.parametrize("wrong_side", [False, True])
def test_frozen_manifest_R_L_side_encoding(data, monkeypatch, wrong_side):
    source = pd.concat([data.development.train.frame, data.development.validation.frame])
    source = source[[*tabular.IDENTITY, *image.IMAGE_COLUMNS]].copy()
    manifest = source.copy()
    manifest["anatomical_side"] = manifest.knee_side_code.map({"1": "R", "2": "L"})
    if wrong_side:
        manifest.loc[manifest.knee_side_code.eq("1"), "anatomical_side"] = "L"
    monkeypatch.setattr(image, "sha256_file", lambda path: image.IMAGE_SHA)
    monkeypatch.setattr(tabular, "load_development", lambda: data.development)
    monkeypatch.setattr(
        image.pd, "read_parquet", lambda path, **kw: manifest if path == image.MANIFEST else source
    )
    if wrong_side:
        with pytest.raises(DatasetIntegrityError, match="Anatomical side"):
            image.load_image_data()
    else:
        loaded = image.load_image_data()
        assert len(loaded.development.train.frame) == 4


def test_integrity_verifies_dtype_shape_SHA_and_never_TEST(data, monkeypatch):
    opened = []
    original = np.load

    def record(path, **kwargs):
        opened.append(path.name)
        assert "LOCKED" not in path.name
        return original(path, **kwargs)

    monkeypatch.setattr(image.np, "load", record)
    audit = image.integrity(data)
    assert len(opened) == 8 and audit["TEST_images_loaded_by_model_loader"] == 0
    assert audit["partitions"]["TRAIN"]["events"] == 2
    assert audit["partitions"]["VALIDATION"]["images"] == 4


@pytest.mark.parametrize("mutation", ["TEST", "overlap", "wrong_side", "duplicate"])
def test_membership_and_leakage_fail_before_image_open(data, config, monkeypatch, mutation):
    p = data.development.train
    if mutation == "TEST":
        p.name = "TEST"
    elif mutation == "overlap":
        p.frame.loc[0, "participant_id"] = "V0"
    elif mutation == "wrong_side":
        p.frame.loc[0, "knee_side_code"] = "3"
    else:
        p.frame.loc[1] = p.frame.iloc[0]
    monkeypatch.setattr(image.np, "load", lambda *a, **k: pytest.fail("Forbidden image opened"))
    with pytest.raises(DatasetIntegrityError):
        image.KneeImages(p, data.root, config, augment=True)


def test_mutated_TEST_membership_blocked_at_getitem(data, config, monkeypatch):
    dataset = image.KneeImages(data.development.train, data.root, config, augment=False)
    dataset.partition.frame.loc[0, "participant_id"] = "LOCKED"
    monkeypatch.setattr(image.np, "load", lambda *a, **k: pytest.fail("TEST image opened"))
    with pytest.raises(DatasetIntegrityError):
        dataset[0]


@pytest.mark.parametrize("mutation", ["missing", "SHA", "dtype", "shape", "escape"])
def test_bad_image_fails_closed(data, config, mutation, tmp_path):
    p = data.development.train
    path = data.root / p.frame.iloc[0].effective_crop_relative_path
    if mutation == "missing":
        p.frame.loc[0, "effective_crop_relative_path"] = "missing.npy"
    elif mutation == "SHA":
        p.frame.loc[0, "effective_crop_sha256"] = "0" * 64
    elif mutation == "escape":
        p.frame.loc[0, "effective_crop_relative_path"] = "../outside.npy"
    else:
        np.save(
            path,
            np.zeros(
                (1067, 1067) if mutation == "dtype" else (32, 32),
                dtype=np.float32 if mutation == "dtype" else np.uint16,
            ),
        )
        p.frame.loc[0, "effective_crop_sha256"] = sha256_file(path)
    with pytest.raises(DatasetIntegrityError):
        image.KneeImages(p, data.root, config, augment=False)[0]


def test_uint16_percentile_channel_and_validation_transform_determinism(data, config):
    dataset = image.KneeImages(data.development.validation, data.root, config, augment=False)
    x, y, _ = dataset[0]
    source = np.load(
        image.resolve_image(data.root, dataset.partition.frame.iloc[0].effective_crop_relative_path)
    )
    original = source.copy()
    a = image.transform_image(source, config)
    assert torch.equal(x, a) and torch.equal(a, image.transform_image(source, config))
    assert a.shape == (3, 320, 320) and a.dtype == torch.float32 and torch.isfinite(a).all()
    recipe = image.WEIGHTS.transforms()
    gray = a * torch.tensor(recipe.std)[:, None, None] + torch.tensor(recipe.mean)[:, None, None]
    torch.testing.assert_close(gray[0], gray[1])
    torch.testing.assert_close(gray[1], gray[2])
    assert gray.min() >= -1e-6 and gray.max() <= 1 + 1e-6
    np.testing.assert_array_equal(source, original)
    assert y.item() == float(dataset.partition.frame.iloc[0][tabular.TARGET])


def test_percentile_policy_matches_existing_frozen_function():
    a = np.arange(1067 * 1067, dtype=np.uint32).reshape(1067, 1067).astype(np.uint16)
    lo, hi = np.percentile(a.astype(np.float32), [0.5, 99.5])
    expected = ((np.clip(a.astype(np.float32), lo, hi) - lo) / (hi - lo)).astype(np.float32)
    np.testing.assert_allclose(image.robust_minmax(a), expected, rtol=1e-6, atol=1e-7)


def test_constant_transform_finite_but_production_integrity_rejects(data, config):
    a = np.full((1067, 1067), 150, dtype=np.uint16)
    assert torch.isfinite(image.transform_image(a, config)).all()
    p = data.development.train
    path = data.root / p.frame.iloc[0].effective_crop_relative_path
    np.save(path, a)
    p.frame.loc[0, "effective_crop_sha256"] = sha256_file(path)
    with pytest.raises(DatasetIntegrityError, match="Degenerate"):
        image.integrity(data)


def test_augmentation_only_TRAIN_deterministic_by_identity_epoch(data, config):
    with pytest.raises(DatasetIntegrityError):
        image.KneeImages(data.development.validation, data.root, config, augment=True)
    t = image.KneeImages(data.development.train, data.root, config, augment=True)
    first = t[0][0]
    assert torch.equal(first, t[0][0])
    t.epoch = 1
    assert not torch.equal(first, t[0][0])
    deterministic = image.KneeImages(t.partition, data.root, config, augment=False)[0][0]
    assert not torch.equal(first, deterministic)
    assert config["augmentation"]["horizontal_flip"] is False


def test_DenseNet_output_BCE_and_backbone_BN_freezing(config):
    torch.set_num_threads(1)
    image.set_seed(config["seed"])
    m = image.build_model(config["candidates"][0], None)
    image.train_mode(m, True)
    assert not m.features.training and m.classifier.training
    assert all(not p.requires_grad for p in m.features.parameters())
    x = torch.ones((1, 3, 320, 320))
    before = {k: v.clone() for k, v in m.features.state_dict().items()}
    y = m(x).flatten()
    assert y.shape == (1,)
    loss = nn.BCEWithLogitsLoss()(y, torch.ones(1))
    assert torch.isfinite(loss)
    loss.backward()
    assert m.classifier.weight.grad is not None
    assert all(torch.equal(v, m.features.state_dict()[k]) for k, v in before.items())


def test_early_stop_best_AUC_and_predeclared_tie_breakers():
    stop = image.EarlyStop(3, 5)

    def score(auc, ap=0.2):
        return {"AUROC": auc, "AUPRC": ap, "Brier": 0.15, "log_loss": 0.4}

    assert stop.observe(1, score(0.6)) == (True, False)
    assert stop.observe(2, score(0.7)) == (True, False)
    assert stop.observe(3, score(0.65)) == (False, False)
    assert stop.observe(4, score(0.7, 0.21)) == (True, False)
    for epoch in (5, 6):
        assert stop.observe(epoch, score(0.69)) == (False, False)
    assert stop.observe(7, score(0.68)) == (False, True)
    assert stop.best_epoch == 4


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 2, 3, stride=16), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(2, 1)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def test_synthetic_fit_checkpoint_predictions_provenance_and_determinism(data, config, tmp_path):
    c = deepcopy(config)
    c["candidates"] = [{"index": 0, "frozen_backbone": False, "lr": 0.001, "epochs": 2}]
    provenance = {
        "source_hashes": data.development.source_hashes,
        "device": image.device_info(torch.device("cpu")),
    }
    original = {str(p): sha256_file(p) for p in data.root.glob("*.npy")}

    def factory(*args):
        return TinyModel()

    first = image.fit_candidate(
        data,
        c,
        c["candidates"][0],
        tmp_path / "unused",
        torch.device("cpu"),
        tmp_path / "run1",
        provenance,
        model_factory=factory,
    )
    second = image.fit_candidate(
        data,
        c,
        c["candidates"][0],
        tmp_path / "unused",
        torch.device("cpu"),
        tmp_path / "run2",
        provenance,
        model_factory=factory,
    )
    np.testing.assert_array_equal(first["probability"], second["probability"])
    saved = torch.load(tmp_path / "run1" / first["checkpoint"], weights_only=True)
    assert saved["epoch"] == first["best_epoch"] and saved["provenance"] == provenance
    assert first["metrics"]["observations"] == 4 and first["metrics"]["positive_events"] == 2
    assert original == {str(p): sha256_file(p) for p in data.root.glob("*.npy")}
    assert not list(tmp_path.rglob("*test_predictions*"))
    wrong = image.KneeImages(data.development.train, data.root, c, augment=False)
    with pytest.raises(DatasetIntegrityError):
        image.validation_probability(TinyModel(), wrong, c, torch.device("cpu"))


def test_validation_prediction_linkage_and_clustered_draws(data):
    p = data.development.validation
    pred = p.frame[[*tabular.IDENTITY, tabular.TARGET]].copy()
    pred["model_id"] = "DenseNet121_selected"
    pred["split"] = "VALIDATION"
    tabular.validate_predictions(p, pred, {"DenseNet121_selected"})
    with pytest.raises((AssertionError, DatasetIntegrityError)):
        tabular.validate_predictions(p, pred.iloc[:-1], {"DenseNet121_selected"})
    for index in tabular.cluster_bootstrap_indices(p, 20, 63027):
        multiplicity = np.bincount(index, minlength=len(p.frame))
        for pid in p.frame.participant_id.unique():
            assert np.unique(multiplicity[p.frame.participant_id.eq(pid)]).size == 1


def test_readiness_reuses_observed_timings_and_checks_provenance(
    data, config, tmp_path, monkeypatch
):
    calls = []
    output = tmp_path / "readiness"
    output.mkdir()

    def sanity(*args):
        calls.append(True)
        (output / "TRAIN_transform_montage.png").write_bytes(b"synthetic montage")
        return {"implementation_sanity_passed": True, "observed_seconds": 1.234}

    monkeypatch.setattr(image, "sanity_and_compute", sanity)
    meta = {"source_hashes": data.development.source_hashes}
    first = image.prepare_readiness(data, config, tmp_path, torch.device("cpu"), output, meta)
    before = sha256_file(output / "readiness_audit.json")
    second = image.prepare_readiness(data, config, tmp_path, torch.device("cpu"), output, meta)
    assert first == second and calls == [True]
    assert before == sha256_file(output / "readiness_audit.json")
    with pytest.raises(DatasetIntegrityError, match="provenance"):
        image.prepare_readiness(
            data, config, tmp_path, torch.device("cpu"), output, {"altered": True}
        )
    (output / "TRAIN_transform_montage.png").write_bytes(b"altered")
    with pytest.raises(DatasetIntegrityError, match="montage"):
        image.prepare_readiness(data, config, tmp_path, torch.device("cpu"), output, meta)
