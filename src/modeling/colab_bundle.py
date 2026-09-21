"""Package exact TRAIN/VALIDATION crops and unchanged 6D training for private Colab T4."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from imaging.artifact_io import file_lock, sha256_file
from modeling import image, tabular
from modeling.colab_assets import runtime
from multimodal.freeze import publish_json, publish_parquet, read_json, require

BUNDLE = image.OUTPUT / "colab_t4_bundle"
ARCHIVE = image.OUTPUT / "colab_t4_bundle.tar.gz"
ASSETS = Path(__file__).parent / "colab_assets"
HELPERS = [
    "DevelopmentPartition",
    "DevelopmentData",
    "calibration",
    "metrics",
    "cluster_bootstrap_indices",
    "bootstrap",
    "validate_predictions",
]
PINS = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "pyarrow": "25.0.1",
    "Pillow": "12.3.0",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
    "matplotlib": "3.11.2",
    "joblib": "1.6.0",
    "threadpoolctl": "3.7.0",
    "narwhals": "2.26.0",
    "filelock": "4.0.0",
    "typing_extensions": "4.16.0",
    "sympy": "1.14.0",
    "networkx": "3.6.1",
    "jinja2": "3.1.6",
    "fsspec": "2026.7.0",
    "setuptools": "84.0.0",
    "mpmath": "1.3.0",
    "markupsafe": "3.0.3",
    "contourpy": "1.4.0",
    "cycler": "0.12.1",
    "fonttools": "4.65.0",
    "kiwisolver": "1.5.1",
    "packaging": "26.3",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "six": "1.17.0",
}


def extraction(source: Path, names: list[str], header: str) -> str:
    """Mechanical verbatim block extraction; never rewrite scientific function bodies."""
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    chunks = []
    found = set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
            chunks.append("".join(lines[first - 1 : node.end_lineno]))
            found.add(node.name)
    require(found == set(names), "Required approved function missing")
    return header + "\n\n" + "\n\n".join(chunks) + "\n"


def copy_exact(source: Path, target: Path, expected: str) -> None:
    require(sha256_file(source) == expected, "Source SHA mismatch before copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(
            sha256_file(target) == expected, "Existing packaged copy differs; refusing overwrite"
        )
    else:
        require(not target.is_symlink(), "Symlink copy target")
        shutil.copyfile(source, target)
    require(sha256_file(target) == expected, "Packaged copy SHA mismatch")
    require(sha256_file(source) == expected, "Source changed during copy")


def generated(path: Path, text: str) -> None:
    """Materialize mechanically generated module/requirements, or validate existing bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, "Generated bundle source differs")
    else:
        path.write_text(text, encoding="utf-8")


def stage_code(root: Path) -> dict:
    for name, value in runtime.CODE.items():
        copy_exact(Path(name), root / name, value)
    path = Path("src/imaging/artifact_io.py")
    copy_exact(path, root / path, sha256_file(path))
    for package in ("modeling", "imaging", "multimodal"):
        generated(root / f"src/{package}/__init__.py", "")
    slim_tabular = extraction(
        Path(tabular.__file__),
        HELPERS,
        '"""Verbatim approved membership, validation metric and bootstrap helpers only."""\n'
        "from __future__ import annotations\n"
        "import hashlib\nfrom dataclasses import dataclass\n"
        "import numpy as np\nimport pandas as pd\n"
        "from scipy.optimize import minimize\nfrom scipy.special import expit\n"
        "from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score\n"
        "from multimodal.freeze import require\n"
        'TARGET = "composite_progression"\n'
        'IDENTITY = ("participant_id", "knee_side_code", "baseline_visit")\n'
        "FEATURE_SETS = {}\n",
    )
    generated(root / "src/modeling/tabular.py", slim_tabular)
    helpers = ["DatasetIntegrityError", "require", "read_json", "publish_json", "publish_parquet"]
    io = extraction(
        Path("src/multimodal/freeze.py"),
        helpers,
        '"""Verbatim artifact I/O guards; no cohort/imaging/review pipeline dependency."""\n'
        "from __future__ import annotations\nimport json\nfrom pathlib import Path\n"
        "import pandas as pd\n"
        "from imaging.artifact_io import atomic_json, atomic_parquet\n",
    )
    # The approved configuration is JSON-compatible YAML; no parser/library substitution.
    io += "\ndef load_config(path):\n    return read_json(Path(path))\n"
    generated(root / "src/multimodal/freeze.py", io)
    for source, destination in (
        ("runtime.py", "src/modeling/bundle_runtime.py"),
        ("verify_bundle.py", "scripts/verify_bundle.py"),
        ("run_training.py", "scripts/run_training.py"),
        ("colab_setup.sh", "scripts/colab_setup.sh"),
        ("README.md", "README.md"),
    ):
        path = ASSETS / source
        copy_exact(path, root / destination, sha256_file(path))
    return {
        "parent_sha256": sha256_file(Path(tabular.__file__)),
        "blocks_sha256": runtime.blocks(Path(tabular.__file__), HELPERS),
    }


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    for name in frame:
        if name == tabular.TARGET:
            frame[name] = frame[name].astype("boolean")
        elif not pd.api.types.is_numeric_dtype(frame[name]):
            frame[name] = frame[name].astype("string")
    return frame


def derive(data: image.ImageData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = []
    for part in (data.development.train, data.development.validation):
        rows = part.frame[[*tabular.IDENTITY, tabular.TARGET, *image.IMAGE_COLUMNS]].copy()
        rows["anatomical_side"] = rows.knee_side_code.map({"1": "R", "2": "L"})
        rows["split"] = part.name
        sources.append(rows[runtime.MANIFEST_COLUMNS])
    manifest = normalize(pd.concat(sources, ignore_index=True))
    directory = tabular.DEFAULT_DIRECTORY / "splits/v1"
    participants = pd.read_parquet(directory / "participant_split_manifest.parquet")
    knees = pd.read_parquet(directory / "knee_split_manifest.parquet")
    people = normalize(
        participants.loc[
            participants.split.isin(runtime.COUNTS),
            [
                "participant_id",
                "split",
                "eligible_knee_count",
                "primary_event_knee_count",
            ],
        ]
    )
    # Match approved per-partition row ordering, not a new random or sorted split.
    selected = pd.concat(
        [
            knees.loc[knees.split.eq(role), [*tabular.IDENTITY, "anatomical_side", "split"]]
            for role in runtime.COUNTS
        ],
        ignore_index=True,
    )
    return manifest, people, normalize(selected)


def build(root: Path = BUNDLE) -> dict:
    for name, expected in runtime.CODE.items():
        require(sha256_file(Path(name)) == expected, "Approved source/config changed")
    require(
        sha256_file(Path(tabular.__file__)) == runtime.METRIC_PARENT_SHA,
        "Approved metric source changed",
    )
    for name, expected in PINS.items():
        require(version(name) == expected, f"Origin dependency changed: {name}")
    baseline = read_json(image.OUTPUT / "pre_change_preservation.json")
    image.preserve(baseline)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "bundle_freeze.json").exists():
        return runtime.verify(root)
    data = image.load_image_data()
    require(data.development.source_hashes == runtime.SOURCES, "Original source hashes changed")
    manifest, people, knees = derive(data)
    for filename, frame in (
        ("image_manifest", manifest),
        ("participant_split_projection", people),
        ("knee_split_projection", knees),
    ):
        publish_parquet(frame, root / f"metadata/{filename}.parquet")
    runtime.tables(root)
    lineage = stage_code(root)
    publish_json(runtime.SOURCES, root / "source_expectations.json")
    publish_json(PINS, root / "dependency_versions.json")
    generated(root / "requirements.txt", "".join(f"{k}=={v}\n" for k, v in sorted(PINS.items())))
    for name in ("preprocessing", "augmentation"):
        original = read_json(image.OUTPUT / f"portable_preparation_v1/{name}_spec.json")
        publish_json(original["specification"], root / f"{name}_spec.json")
    publish_json(
        {"version": runtime.VERSION, "source_hashes": runtime.SOURCES},
        root / "data/processed/modeling/image/v1/pre_change_preservation.json",
    )
    weights = image.OUTPUT / "pretrained/densenet121-a639ec97.pth"
    copy_exact(weights, root / "pretrained/densenet121-a639ec97.pth", runtime.WEIGHT_SHA)
    for number, row in enumerate(manifest.itertuples(index=False), 1):
        source = image.resolve_image(data.root, row.effective_crop_relative_path)
        destination = root / "images" / row.effective_crop_relative_path
        require(
            destination.resolve().is_relative_to((root / "images").resolve()), "Copy path escape"
        )
        copy_exact(source, destination, row.effective_crop_sha256)
        if number % 1000 == 0:
            print(f"Copied/verified {number}/5917 effective crops", flush=True)
    wanted = {"images/" + p for p in manifest.effective_crop_relative_path}
    actual = {p.relative_to(root).as_posix() for p in (root / "images").rglob("*") if p.is_file()}
    require(wanted == actual, "Unexpected/missing packaged image")
    image.preserve(baseline)
    catalog = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    freeze = {
        "version": runtime.VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "derived_from_authoritative_frozen_sources": True,
        "original_full_source_hashes_recomputed_on_Mac": True,
        "original_sources_not_shipped": True,
        "source_hashes": runtime.SOURCES,
        "approved_code_hashes": runtime.CODE,
        "verbatim_metrics": lineage,
        "artifacts_sha256": catalog,
        "TRAIN_images": 4873,
        "VALIDATION_images": 1044,
        "TEST_images": 0,
        "pretrained_identifier": "DenseNet121_Weights.IMAGENET1K_V1",
        "pretrained_sha256": runtime.WEIGHT_SHA,
        "origin_preservation_audit_sha256": sha256_file(
            image.OUTPUT / "portable_implementation_audit.json"
        ),
        "origin_protected_artifact_preservation_passed": True,
        "no_model_fit_or_inference_during_packaging": True,
    }
    publish_json(freeze, root / "bundle_freeze.json")
    return runtime.verify(root)


def archive(root: Path, path: Path) -> dict:
    verification = runtime.verify(root)
    report_path = path.with_suffix(path.suffix + ".json")
    if path.exists():
        report = read_json(report_path)
        require(sha256_file(path) == report["archive_sha256"], "Existing transfer archive changed")
        require(
            sha256_file(root / "bundle_freeze.json") == report["bundle_freeze_sha256"],
            "Archive bundle changed",
        )
        return report
    temporary = path.with_name(path.name + ".partial")
    require(not temporary.exists(), "Interrupted archive retained; refusing overwrite")
    # Deterministic ordering, zero timestamps/owners and stable gzip header.
    with temporary.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        entry = tar.gettarinfo(
                            str(p), arcname="colab_t4_bundle/" + p.relative_to(root).as_posix()
                        )
                        entry.mtime, entry.uid, entry.gid = 0, 0, 0
                        entry.uname, entry.gname = "", ""
                        entry.mode = 0o644
                        entry.pax_headers = {}
                        with p.open("rb") as stream:
                            tar.addfile(entry, stream)
    temporary.replace(path)
    # Independently stream every archived member and compare its exact payload SHA.
    expected = {
        p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()
    }
    found = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            require(
                member.isfile() and member.name.startswith("colab_t4_bundle/"),
                "Unsafe archive member",
            )
            relative = member.name.removeprefix("colab_t4_bundle/")
            require(
                relative in expected and relative not in found,
                "Unexpected/duplicate archive member",
            )
            payload = tar.extractfile(member)
            require(payload is not None, "Archive payload missing")
            digest = hashlib.sha256()
            for block in iter(lambda stream=payload: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
            found[relative] = digest.hexdigest()
    require(found == expected, "Archived bundle payload SHA mismatch")
    report = {
        "archive_sha256": sha256_file(path),
        "archive_bytes": path.stat().st_size,
        "bundle_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "bundle_files": len(expected),
        "bundle_freeze_sha256": sha256_file(root / "bundle_freeze.json"),
        "TRAIN_images": verification["TRAIN"],
        "VALIDATION_images": verification["VALIDATION"],
        "TEST_images": 0,
        "archive_member_payload_SHA_verification_passed": True,
    }
    publish_json(report, report_path)
    return report


def relocation(root: Path) -> dict:
    moved = root.with_name("colab_t4_bundle_relocation_check")
    require(not moved.exists(), "Relocation destination already exists")
    root.rename(moved)
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        # Remove editable-repository/import overrides: only the packaged source is used.
        env.pop("PYTHONPATH", None)
        outputs = []
        for script, arguments in (("verify_bundle.py", []), ("run_training.py", ["--plan"])):
            result = subprocess.run(
                [sys.executable, str((moved / "scripts" / script).resolve()), *arguments],
                cwd="/private/tmp",
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.append(json.loads(result.stdout))
        require(outputs[1]["fit_performed"] is False, "Relocation performed a fit")
        return {
            "relocated_verification": outputs[0],
            "production_command_resolves": True,
            "fit_performed": False,
        }
    finally:
        moved.rename(root)


def validate_return(root: Path, output: Path) -> dict:
    runtime.verify(root)
    image.preserve(read_json(image.OUTPUT / "pre_change_preservation.json"))
    require(
        output.resolve() != (root / runtime.PRODUCTION).resolve(),
        "Validate downloaded staging output, not live bundle output",
    )
    frozen = read_json(output / "run_freeze.json")
    require(
        frozen["status"] == "FROZEN_IMAGE_DEVELOPMENT" and frozen["TEST_scored"] is False,
        "Returned model is not a valid development freeze",
    )
    require(frozen["TEST_images_loaded_by_model_loader"] == 0, "Returned model loaded TEST")
    meta = frozen["provenance"]
    require(
        meta["validation_metric_code_sha256"] == sha256_file(root / "src/modeling/tabular.py")
        and meta["normalization_source_sha256"] == runtime.CODE["src/imaging/preprocessing.py"],
        "Returned scientific helper source differs",
    )
    require(
        meta["portable_execution"]["adapter_sha256"]
        == sha256_file(root / "src/modeling/bundle_runtime.py")
        and meta["portable_execution"]["original_metrics_source_sha256"]
        == runtime.METRIC_PARENT_SHA,
        "Returned adapter/helper lineage differs",
    )
    require(meta["source_hashes"] == runtime.SOURCES, "Returned source lineage differs")
    require(
        meta["training_code_sha256"] == runtime.CODE["src/modeling/image.py"],
        "Returned training source differs",
    )
    require(
        meta["model_configuration_sha256"] == runtime.CODE["configs/image_model_v1.yaml"],
        "Returned configuration differs",
    )
    require(meta["pretrained_sha256"] == runtime.WEIGHT_SHA, "Returned pretrained weights differ")
    require(
        meta["portable_execution"]["bundle_freeze_sha256"]
        == sha256_file(root / "bundle_freeze.json"),
        "Returned bundle lineage differs",
    )
    require(
        meta["device"]["device"] == "cuda"
        and runtime.approved_gpu_priority(meta["device"]["CUDA_name"]) is not None,
        "Returned production did not use approved A100/L4/T4 CUDA",
    )
    require(
        (meta["TRAIN_count"], meta["VALIDATION_count"]) == (4873, 1044), "Returned census differs"
    )
    catalog = frozen["artifacts_sha256"]
    actual = set()
    for p in output.rglob("*"):
        require(not p.is_symlink(), "Returned artifact symlink")
        if p.is_file() and p.name != "run_freeze.json.lock":
            actual.add(p.relative_to(output).as_posix())
    require(actual == set(catalog) | {"run_freeze.json"}, "Unexpected/missing returned artifact")
    for name, value in catalog.items():
        require(
            sha256_file(runtime.safe_path(output, name)) == value, "Returned artifact SHA mismatch"
        )
    data = image.load_image_data()
    manifest, _, _ = runtime.tables(root)
    expected = manifest.loc[manifest.split.eq("VALIDATION")].reset_index(drop=True)
    original = data.development.validation.frame
    for c in (*tabular.IDENTITY, tabular.TARGET):
        require(
            expected[c].tolist() == original[c].tolist(),
            "Returned bundle/original validation linkage differs",
        )
    membership = (
        manifest.drop_duplicates("participant_id").set_index("participant_id").split.to_dict()
    )
    valid = tabular.DevelopmentPartition(
        "VALIDATION",
        expected,
        membership,
        frozenset(expected[list(tabular.IDENTITY)].itertuples(index=False, name=None)),
    )
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    tabular.validate_predictions(valid, predictions, {"DenseNet121_selected"})
    p = predictions.predicted_probability.to_numpy()
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid returned probabilities")
    config = read_json(root / "configs/image_model_v1.yaml")
    image.validate_config(config)
    preprocessing, augmentation = image.specifications(config)
    require(
        meta["preprocessing_sha256"] == image.digest(preprocessing)
        and meta["augmentation_sha256"] == image.digest(augmentation),
        "Returned transform specifications differ",
    )
    require(
        (meta["seed"], meta["bootstrap_seed"]) == (config["seed"], config["bootstrap_seed"]),
        "Returned seeds differ",
    )
    require(
        meta["DataLoader_seed"] == meta["augmentation_seed"] == config["seed"],
        "Returned data/augmentation seeds differ",
    )
    for name, pin in (("torch", PINS["torch"]), ("torchvision", PINS["torchvision"])):
        require(meta["device"][name].split("+")[0] == pin, "Returned library version differs")
    selected = read_json(output / "selected_candidate.json")
    candidates = read_json(output / "candidate_results.json")
    require(
        candidates["provenance"] == meta and len(candidates["candidates"]) == 3,
        "Returned candidate catalog differs",
    )
    import torch

    for candidate, result in zip(config["candidates"], candidates["candidates"], strict=True):
        require(result["candidate"] == candidate, "Returned predeclared candidate differs")
        history = (
            read_json(output / "candidate_0/input_integrity.json")
            if candidate["index"] == 0
            else read_json(output / f"candidate_{candidate['index']}/input_integrity.json")
        )
        require(history["provenance"] == meta, "Returned per-fit lineage differs")
        with (output / f"candidate_{candidate['index']}/history.json").open() as stream:
            epochs = json.load(stream)
        require(
            len(epochs) == result["epochs_run"] <= candidate["epochs"],
            "Returned epoch accounting differs",
        )
        require(
            [row["epoch"] for row in epochs] == list(range(1, len(epochs) + 1)),
            "Returned epoch ordering differs",
        )
        stopping = image.EarlyStop(config["minimum_epochs"], config["patience"])
        stops = []
        for row in epochs:
            require(np.isfinite(row["TRAIN_BCE"]), "Returned training loss nonfinite")
            stops.append(stopping.observe(row["epoch"], row["VALIDATION"])[1])
        require(not any(stops[:-1]), "Returned training continued after early stopping")
        require(
            result["early_stopped"] == (len(epochs) < candidate["epochs"]),
            "Returned early-stop flag differs",
        )
        require(not result["early_stopped"] or stops[-1], "Returned training ended prematurely")
        best = min(epochs, key=lambda r: image.selection_key(r["VALIDATION"]))
        require(
            best["epoch"] == result["best_epoch"],
            "Returned checkpoint is not best validation epoch",
        )
        state = torch.load(output / result["checkpoint"], map_location="cpu", weights_only=True)
        require(
            state["candidate"] == candidate
            and state["epoch"] == result["best_epoch"]
            and state["provenance"] == meta,
            "Returned checkpoint provenance differs",
        )
        require(
            state["validation_metrics"] == result["metrics"], "Returned checkpoint metrics differ"
        )
        require(
            sha256_file(output / result["checkpoint"]) == result["checkpoint_sha256"],
            "Returned checkpoint SHA differs",
        )
    best = min(
        candidates["candidates"],
        key=lambda r: image.selection_key(r["metrics"], r["candidate"]["index"]),
    )
    require(
        {k: selected[k] for k in best} == best and selected["provenance"] == meta,
        "Returned selection objective differs",
    )
    score = tabular.metrics(valid, p)
    for c in ("AUROC", "AUPRC", "Brier", "log_loss"):
        require(
            abs(score[c] - selected["metrics"][c]) < 1e-12, "Returned validation metric mismatch"
        )
    require(
        read_json(output / "validation_metrics.json")
        == {"provenance": meta, "metrics": selected["metrics"]},
        "Returned selected metric catalog differs",
    )
    for c in ("intercept", "slope"):
        actual, recorded = score["calibration"][c], selected["metrics"]["calibration"][c]
        require(
            (actual is None and recorded is None)
            or (actual is not None and recorded is not None and abs(actual - recorded) < 1e-6),
            "Returned calibration diagnostic differs",
        )
    boot, intervals = tabular.bootstrap(valid, {"DenseNet121_selected": p}, config)
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "bootstrap_metrics.parquet"),
        boot,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )
    recorded_boot = read_json(output / "clustered_bootstrap.json")
    require(recorded_boot.pop("provenance") == meta, "Returned bootstrap lineage differs")
    recorded_intervals = recorded_boot.pop("intervals")
    require(
        recorded_boot == {k: v for k, v in intervals.items() if k != "intervals"},
        "Returned bootstrap cluster/draw policy differs",
    )
    for name, group in intervals["intervals"].items():
        for metric, values in group.items():
            for key, value in values.items():
                require(
                    abs(value - recorded_intervals[name][metric][key]) < 1e-12,
                    "Returned clustered bootstrap interval differs",
                )
    from sklearn.calibration import calibration_curve

    actual, predicted = calibration_curve(valid.labels(), p, n_bins=10, strategy="quantile")
    curve = read_json(output / "calibration_data.json")
    require(
        curve["partition"] == "VALIDATION" and curve["recalibration_applied"] is False,
        "Wrong calibration role",
    )
    np.testing.assert_allclose(actual, curve["event_fraction"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(predicted, curve["mean_probability"], rtol=0, atol=1e-12)
    return {
        "return_validation_passed": True,
        "validation_rows": len(predictions),
        "TEST_scored": False,
        "fit_or_image_inference_performed": False,
        "local_import_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "validate-return"))
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "validate-return":
        require(args.output is not None, "Downloaded production output path required")
        result = validate_return(args.bundle, args.output)
    else:
        with file_lock(image.OUTPUT / "colab_bundle_build"):
            audit = build(args.bundle)
            relocated = relocation(args.bundle)
            result = {
                "verification": audit,
                "relocation": relocated,
                "transfer": archive(args.bundle, ARCHIVE),
            }
        publish_json(result, image.OUTPUT / "colab_bundle_audit.json")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
