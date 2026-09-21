"""Contract tests for public release assets and aggregate scientific artifacts.

Validates that publicly distributed assets (code, configs, figures, documentation,
and aggregate scientific JSON summaries) exist, conform to expected schemas, match
authoritative cryptographic digests, and are mathematically consistent with documented
metrics, without requiring access to raw data or private participant records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imaging.artifact_io import sha256_file

pytestmark = pytest.mark.public_portable

EXPECTED_SCORING_CODE_SHA = "fc18e711254c9b9d97dc38327bb0d101f36955bccb8fbc5930480231e3343a34"
EXPECTED_ROC_PLOT_SHA = "9e90ba58d282fbedd9c2199df26811691165ebbd9a78274dc8f35de59c26ae3e"
EXPECTED_PR_PLOT_SHA = "3e3ba755168c26219c134fd4217bb019ecd314bf0334dd7fc830bbf9a91e9c13"
EXPECTED_CAL_PLOT_SHA = "78e6865ff4441baeb495cb4b5b08dab621c07318413380d185c0a6e7bf5cac64"

EXPECTED_AGGREGATE_SHAS = {
    "test_metrics.json": "d001a21938e5406a440dd82ea883acdf1f088a6ebb3999743cd29f2e8ad63702",
    "clustered_bootstrap.json": "67d493173c2f50091c7e2ce22ec643d25afbd1ae87dddbd4c55d80eeee7064ec",
    "paired_bootstrap_differences.json": "ed0eef81461ebac4693bc85faecc74a31ae47262917bd88dadbdd2d7c993f0d0",
    "calibration_data.json": "b99c74e7bcc1cb6880d91acbb0ed369377db497b3feaf7ac9efabb7f50caca06",
    "validation_vs_test_summary.json": "0baed5db4ea8f20b44029956ba93042eb059b8e3c131e2e2df592ceefc15f447",
}

RESULTS_DIR = Path("docs/results")
ASSETS_DIR = Path("docs/assets")


def test_frozen_scoring_code_sha():
    """Scoring runner must match the immutable post-evaluation frozen SHA-256."""
    historical_txt = Path("docs/historical/final_test_evaluation.py.txt")
    private_runner = Path("src/modeling/final_test_evaluation.py")

    target = historical_txt if historical_txt.is_file() else private_runner
    assert target.is_file(), (
        f"Neither historical snapshot ({historical_txt}) nor private runner ({private_runner}) exists"
    )
    assert sha256_file(target) == EXPECTED_SCORING_CODE_SHA

    # If both exist (in local research repo), ensure both are byte-identical
    if historical_txt.is_file() and private_runner.is_file():
        assert sha256_file(historical_txt) == sha256_file(private_runner)


def _parse_markdown_tables(path: Path) -> list[list[dict[str, str]]]:
    """Extract and parse all markdown tables in a file into lists of row dictionaries."""
    import re

    content = path.read_text(encoding="utf-8")
    tables: list[list[list[str]]] = []
    current_table: list[list[str]] = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [re.sub(r"\\?\*+", "", c).strip() for c in stripped.split("|")[1:-1]]
            current_table.append(cells)
        else:
            if len(current_table) >= 2:
                tables.append(current_table)
            current_table = []

    if len(current_table) >= 2:
        tables.append(current_table)

    parsed_tables: list[list[dict[str, str]]] = []
    for raw in tables:
        headers = raw[0]
        rows = [r for r in raw[1:] if not all(re.match(r"^:?-+:?$", c) for c in r)]
        parsed_table: list[dict[str, str]] = []
        for r in rows:
            row_dict = {h: cell for h, cell in zip(headers, r, strict=True)}
            parsed_table.append(row_dict)
        parsed_tables.append(parsed_table)

    return parsed_tables


def test_public_aggregate_figures_hashes():
    """Diagnostic held-out evaluation plots must match frozen digests byte-for-byte."""
    assert ASSETS_DIR.is_dir(), "Missing docs/assets directory"
    assert sha256_file(ASSETS_DIR / "roc_curve.png") == EXPECTED_ROC_PLOT_SHA
    assert sha256_file(ASSETS_DIR / "precision_recall_curve.png") == EXPECTED_PR_PLOT_SHA
    assert sha256_file(ASSETS_DIR / "calibration_curve.png") == EXPECTED_CAL_PLOT_SHA


def test_public_aggregate_results_hashes():
    """All public aggregate JSON summaries must match verified digests byte-for-byte."""
    assert RESULTS_DIR.is_dir(), "Missing docs/results directory"
    for fname, expected_sha in EXPECTED_AGGREGATE_SHAS.items():
        p = RESULTS_DIR / fname
        assert p.is_file(), f"Missing aggregate result file: {p}"
        assert sha256_file(p) == expected_sha, f"SHA mismatch for {fname}"


def test_errata_artifacts_exist_and_match():
    """Authoritative errata files must exist, agree contextually, and document Fusion B validation discrepancy."""
    errata_md = RESULTS_DIR / "ERRATA.md"
    errata_json = RESULTS_DIR / "errata.json"
    assert errata_md.is_file(), f"Missing {errata_md}"
    assert errata_json.is_file(), f"Missing {errata_json}"

    # 1. Verify errata.json structure and authoritative values
    data = json.loads(errata_json.read_text(encoding="utf-8"))
    assert data["status"] == "PRESERVED_ARCHIVAL_ARTIFACT_WITH_DOCUMENTARY_ERRATUM"
    assert data["target_artifact"] == "docs/results/validation_vs_test_summary.json"
    assert data["impact_on_test_results"] is False
    assert data["recomputed_metrics"] is False

    auth = data["authoritative_fusion_b_validation_metrics"]
    assert auth["model_id"] == "multimodal_fusion_B_sensitivity"
    assert auth["partition"] == "VALIDATION"
    assert auth["selected_alpha_image"] == 0.60
    assert auth["AUROC"] == 0.6948924825728214
    assert auth["AUPRC"] == 0.29928367626042385
    assert auth["Brier"] == 0.1227010693382655

    # 2. Contextually parse ERRATA.md markdown table
    md_tables = _parse_markdown_tables(errata_md)
    assert len(md_tables) >= 1, "Discrepancy table missing in ERRATA.md"
    errata_rows = {
        row["Field"].replace("*", "").strip(): row for row in md_tables[0] if "Field" in row
    }

    val_col = "Authoritative Frozen Value (`selected_fusion.json` & `final_model_freeze.json`)"
    assert "0.60" in errata_rows["Selected $\\alpha$ (Image Weight)"][val_col]
    assert "0.694892" in errata_rows["Validation AUROC"][val_col]
    assert "0.6948924825728214" in errata_rows["Validation AUROC"][val_col]
    assert "0.299284" in errata_rows["Validation AUPRC"][val_col]
    assert "0.29928367626042385" in errata_rows["Validation AUPRC"][val_col]
    assert "0.122701" in errata_rows["Validation Brier Score"][val_col]
    assert "0.12270106933826550" in errata_rows["Validation Brier Score"][val_col]

    # 3. Verify narrative context in ERRATA.md associates model and partition
    md_text = errata_md.read_text(encoding="utf-8")
    assert "Multimodal Fusion B" in md_text
    assert "VALIDATION" in md_text
    assert "0.694892" in md_text
    assert "0.735149" in md_text  # TEST AUROC remains unaffected


def test_public_aggregate_results_schemas():
    """Aggregate result JSON files must conform to expected schema and contain no participant data."""
    for fname in EXPECTED_AGGREGATE_SHAS:
        content = (RESULTS_DIR / fname).read_text(encoding="utf-8")
        assert "/Users/" not in content, f"Local user path found in {fname}"
        assert "/home/" not in content, f"System home path found in {fname}"
        data = json.loads(content)
        if fname == "test_metrics.json":
            assert "primary" in data and "sensitivity" in data
            assert "multimodal_fusion_A_selected" in data["primary"]
            assert "DenseNet121_selected" in data["primary"]
            assert "logistic__formulation_A" in data["primary"]
        elif fname == "clustered_bootstrap.json":
            assert data["replicates"] == 1000
            assert data["seed"] == 65027
            assert "intervals" in data
        elif fname == "paired_bootstrap_differences.json":
            assert "comparisons" in data
            assert "fusion_minus_image" in data["comparisons"]
            assert "fusion_minus_logistic_A" in data["comparisons"]
        elif fname == "calibration_data.json":
            assert data["partition"] == "TEST"
            assert data["recalibration_applied"] is False
        elif fname == "validation_vs_test_summary.json":
            assert isinstance(data, list)
            models = {item["model"] for item in data}
            assert "DenseNet121 (Primary Image)" in models
            assert "Multimodal Fusion A (Primary)" in models


def test_markdown_tables_census_and_splits():
    """Contextually parsed markdown census tables must match authoritative split counts."""
    # 1. README census table
    readme_tables = _parse_markdown_tables(Path("README.md"))
    census_table = next(t for t in readme_tables if any("Partition" in row for row in t))
    census_by_part = {row["Partition"].replace("*", "").strip(): row for row in census_table}

    assert census_by_part["TRAIN"]["Knees ($N$)"] == "4,873"
    assert census_by_part["TRAIN"]["Unique Participants"] == "2,535"
    assert census_by_part["TRAIN"]["Progression Events ($Y=1$)"] == "749"
    assert census_by_part["TRAIN"]["Event Rate (%)"] == "15.370%"

    assert census_by_part["VALIDATION"]["Knees ($N$)"] == "1,044"
    assert census_by_part["VALIDATION"]["Unique Participants"] == "543"
    assert census_by_part["VALIDATION"]["Progression Events ($Y=1$)"] == "161"
    assert census_by_part["VALIDATION"]["Event Rate (%)"] == "15.421%"

    assert census_by_part["TEST"]["Knees ($N$)"] == "1,044"
    assert census_by_part["TEST"]["Unique Participants"] == "543"
    assert census_by_part["TEST"]["Progression Events ($Y=1$)"] == "161"
    assert census_by_part["TEST"]["Event Rate (%)"] == "15.421%"

    assert census_by_part["Full Cohort"]["Knees ($N$)"] == "6,961"
    assert census_by_part["Full Cohort"]["Unique Participants"] == "3,621"
    assert census_by_part["Full Cohort"]["Progression Events ($Y=1$)"] == "1,071"
    assert census_by_part["Full Cohort"]["Event Rate (%)"] == "15.386%"

    # 2. cohort_and_outcomes.md census table
    co_tables = _parse_markdown_tables(Path("docs/cohort_and_outcomes.md"))
    co_census = {row["Attribute"].replace("*", "").strip(): row for row in co_tables[0]}
    assert co_census["Total Knees ($N$)"]["TRAIN"] == "4,873"
    assert co_census["Total Knees ($N$)"]["VALIDATION"] == "1,044"
    assert co_census["Total Knees ($N$)"]["TEST"] == "1,044"
    assert co_census["Total Knees ($N$)"]["Full Cohort"] == "6,961"
    assert co_census["Unique Participants"]["TRAIN"] == "2,535"
    assert co_census["Unique Participants"]["VALIDATION"] == "543"
    assert co_census["Unique Participants"]["TEST"] == "543"
    assert co_census["Unique Participants"]["Full Cohort"] == "3,621"
    assert co_census["Composite Progression Events ($Y=1$)"]["Full Cohort"] == "1,071"


def test_markdown_tables_results_and_generalization():
    """Contextually parsed results and generalization tables must match aggregate values."""
    # 1. README results table
    readme_tables = _parse_markdown_tables(Path("README.md"))
    results_table = next(t for t in readme_tables if any("Model Architecture" in row for row in t))
    res_by_model = {
        row["Model Architecture"].replace("*", "").strip(): row for row in results_table
    }

    assert res_by_model["Logistic Formulation A"]["AUROC"] == "0.644"
    assert res_by_model["Logistic Formulation A"]["AUPRC"] == "0.236"
    assert res_by_model["Logistic Formulation A"]["Brier Score"] == "0.127"
    assert res_by_model["Logistic Formulation A"]["Log Loss"] == "0.415"

    assert res_by_model["DenseNet121 Candidate 2"]["AUROC"] == "0.732"
    assert res_by_model["DenseNet121 Candidate 2"]["AUPRC"] == "0.371"
    assert res_by_model["DenseNet121 Candidate 2"]["Brier Score"] == "0.118"
    assert res_by_model["DenseNet121 Candidate 2"]["Log Loss"] == "0.391"

    assert res_by_model["Primary Multimodal Fusion A"]["AUROC"] == "0.728"
    assert res_by_model["Primary Multimodal Fusion A"]["AUPRC"] == "0.368"
    assert res_by_model["Primary Multimodal Fusion A"]["Brier Score"] == "0.118"
    assert res_by_model["Primary Multimodal Fusion A"]["Log Loss"] == "0.387"

    # 2. docs/final_results.md tables
    fr_tables = _parse_markdown_tables(Path("docs/final_results.md"))

    # Table 0: Primary Model Performance
    prim_rows = {row["Evaluation Metric"].replace("*", "").strip(): row for row in fr_tables[0]}
    assert "0.643582" in prim_rows["AUROC [95% CI]"]["Logistic Formulation A (Primary Tabular)"]
    assert "0.731963" in prim_rows["AUROC [95% CI]"]["DenseNet121 Candidate 2 (Primary Image)"]
    assert (
        "0.728164" in prim_rows["AUROC [95% CI]"]["Multimodal Fusion A ($\\alpha = 0.50$, Primary)"]
    )

    assert "0.236237" in prim_rows["AUPRC [95% CI]"]["Logistic Formulation A (Primary Tabular)"]
    assert "0.371287" in prim_rows["AUPRC [95% CI]"]["DenseNet121 Candidate 2 (Primary Image)"]
    assert (
        "0.367969" in prim_rows["AUPRC [95% CI]"]["Multimodal Fusion A ($\\alpha = 0.50$, Primary)"]
    )

    assert (
        "0.126882" in prim_rows["Brier Score [95% CI]"]["Logistic Formulation A (Primary Tabular)"]
    )
    assert (
        "0.117505" in prim_rows["Brier Score [95% CI]"]["DenseNet121 Candidate 2 (Primary Image)"]
    )
    assert (
        "0.117951"
        in prim_rows["Brier Score [95% CI]"]["Multimodal Fusion A ($\\alpha = 0.50$, Primary)"]
    )

    # Table 1: Sensitivity Analyses
    sens_rows = {row["Evaluation Metric"].replace("*", "").strip(): row for row in fr_tables[1]}
    assert "0.685667" in sens_rows["AUROC"]["Logistic Formulation B (+ Baseline KL)"]
    assert "0.735149" in sens_rows["AUROC"]["Multimodal Fusion B ($\\alpha = 0.60$)"]
    assert "0.269416" in sens_rows["AUPRC"]["Logistic Formulation B (+ Baseline KL)"]
    assert "0.373690" in sens_rows["AUPRC"]["Multimodal Fusion B ($\\alpha = 0.60$)"]
    assert "0.123897" in sens_rows["Brier Score"]["Logistic Formulation B (+ Baseline KL)"]
    assert "0.116845" in sens_rows["Brier Score"]["Multimodal Fusion B ($\\alpha = 0.60$)"]

    # Table 2: Generalization (Val vs TEST)
    gen_rows = {row["Model"].replace("*", "").strip(): row for row in fr_tables[2]}
    assert gen_rows["Logistic A"]["Val AUROC"] == "0.651692"
    assert gen_rows["Logistic A"]["TEST AUROC"] == "0.643582"
    assert gen_rows["Logistic A"]["$\\Delta$ AUROC"] == "-0.008110"

    assert gen_rows["DenseNet121"]["Val AUROC"] == "0.677856"
    assert gen_rows["DenseNet121"]["TEST AUROC"] == "0.731963"
    assert gen_rows["DenseNet121"]["$\\Delta$ AUROC"] == "+0.054107"

    assert gen_rows["Fusion A"]["Val AUROC"] == "0.700105"
    assert gen_rows["Fusion A"]["TEST AUROC"] == "0.728164"
    assert gen_rows["Fusion A"]["$\\Delta$ AUROC"] == "+0.028059"

    assert gen_rows["Logistic B"]["Val AUROC"] == "0.663953"
    assert gen_rows["Logistic B"]["TEST AUROC"] == "0.685667"
    assert gen_rows["Logistic B"]["$\\Delta$ AUROC"] == "+0.021714"

    assert gen_rows["Fusion B (Archival)"]["Val AUROC"] == "0.703276"
    assert gen_rows["Fusion B (Archival)"]["TEST AUROC"] == "0.735149"
    assert gen_rows["Fusion B (Archival)"]["$\\Delta$ AUROC"] == "+0.031873"


def test_feature_rosters_and_exclusions():
    """Verify authoritative feature rosters and absence of false claims (400m walk, missingness indicators)."""
    import re

    modeling_doc = Path("docs/modeling_and_evaluation.md").read_text(encoding="utf-8")
    assert "Predictor Set (14 features):" in modeling_doc
    assert "Predictor Set (15 features):" in modeling_doc

    expected_14_features = [
        "age_years",
        "sex",
        "bmi",
        "prior_knee_surgery",
        "family_knee_replacement_history",
        "womac_pain",
        "womac_stiffness",
        "womac_disability",
        "koos_pain",
        "koos_symptoms",
        "walk_20m_pace_mps",
        "chair_stand_time_seconds",
        "knee_extension_strength_n",
        "knee_flexion_strength_n",
    ]
    expected_15_features = [*expected_14_features, "baseline_kl"]

    # 1. Parse numbered list in docs/modeling_and_evaluation.md for Formulation A
    parsed_a = re.findall(r"^\s*\d+\.\s+`([a-z0-9_]+)`", modeling_doc, re.MULTILINE)
    assert parsed_a == expected_14_features, (
        f"Formulation A roster mismatch! Parsed: {parsed_a} vs Expected: {expected_14_features}"
    )

    # 2. Assert Formulation B explicitly defines the exact 15 features (Formulation A + baseline_kl)
    assert "Identical to Formulation A, plus `baseline_kl`" in modeling_doc
    # Verify set equality and exact difference is {'baseline_kl'}
    set_a = set(expected_14_features)
    set_b = set(expected_15_features)
    assert len(set_a) == 14
    assert len(set_b) == 15
    assert set_b - set_a == {"baseline_kl"}
    assert set_a - set_b == set()

    # 3. Verify code constants match authoritative rosters exactly
    from multimodal.freeze import CLINICAL, FUNCTION, PRO

    code_formulation_a = list((*CLINICAL, *PRO, *FUNCTION))
    assert code_formulation_a == expected_14_features, (
        f"Code constant mismatch for Formulation A: {code_formulation_a}"
    )
    code_formulation_b = [*code_formulation_a, "baseline_kl"]
    assert code_formulation_b == expected_15_features, (
        f"Code constant mismatch for Formulation B: {code_formulation_b}"
    )

    # 4. Verify no 400m walk and no missingness indicators claims across core docs
    core_docs = [
        Path("README.md"),
        Path("docs/project_overview.md"),
        Path("docs/methodology.md"),
        Path("docs/cohort_and_outcomes.md"),
        Path("docs/modeling_and_evaluation.md"),
        Path("docs/portfolio_summary.md"),
    ]
    for doc in core_docs:
        text = doc.read_text(encoding="utf-8")
        assert "400-meter" not in text and "400m" not in text, f"400m walk mentioned in {doc}"
        # Ensure no affirmative claims that missingness indicators were used or added
        assert "with missingness indicator" not in text.lower(), (
            f"Affirmative missingness indicator claim in {doc}"
        )
        assert "missingness indicators were used" not in text.lower(), (
            f"Affirmative missingness indicator claim in {doc}"
        )
        if "missingness indicator" in text.lower():
            # If mentioned, it must be explicitly negated
            assert (
                "no missingness indicator" in text.lower()
                or "no added missingness indicator" in text.lower()
            ), f"Missingness indicator mentioned without explicit negation in {doc}"


def test_public_documentation_files_exist_and_nonempty():
    """All core public documentation files must exist and be non-empty."""
    public_docs = [
        "README.md",
        "data/README.md",
        "docs/project_overview.md",
        "docs/methodology.md",
        "docs/cohort_and_outcomes.md",
        "docs/imaging_pipeline.md",
        "docs/modeling_and_evaluation.md",
        "docs/final_results.md",
        "docs/limitations.md",
        "docs/reproducibility.md",
        "docs/data_access_and_privacy.md",
        "docs/repository_structure.md",
        "docs/portfolio_summary.md",
        "docs/results/README.md",
        "docs/results/ERRATA.md",
        "docs/historical/README.md",
    ]
    for doc in public_docs:
        p = Path(doc)
        assert p.is_file(), f"Missing expected public document: {doc}"
        assert p.stat().st_size > 100, f"Public document too small: {doc}"


def test_narrative_and_document_validation_vs_test_consistency():
    """Verify narrative and tabular validation-vs-TEST metrics match aggregate JSON and reject wrong values."""
    val_summary_path = RESULTS_DIR / "validation_vs_test_summary.json"
    assert val_summary_path.is_file(), f"Missing {val_summary_path}"
    summary = json.loads(val_summary_path.read_text(encoding="utf-8"))

    dense_entry = next((item for item in summary if "DenseNet121" in item["model"]), None)
    assert dense_entry is not None, "DenseNet121 missing in validation_vs_test_summary.json"

    val_auroc = dense_entry["val_AUROC"]
    test_auroc = dense_entry["test_AUROC"]
    diff_auroc = dense_entry["diff_AUROC"]

    # Assert exact authoritative values from aggregate JSON
    assert val_auroc == 0.677856
    assert test_auroc == 0.731963
    assert diff_auroc == 0.054107

    # Documents that state validation vs test image discrimination narrative
    target_docs = [Path("README.md"), Path("docs/limitations.md"), Path("docs/final_results.md")]
    for doc in target_docs:
        assert doc.is_file(), f"Missing target document: {doc}"
        text = doc.read_text(encoding="utf-8")

        # Must reject incorrect / unaligned values
        assert "0.729" not in text, f"Incorrect validation AUROC 0.729 found in {doc}"
        assert "0.598" not in text, f"Incorrect test AUROC 0.598 found in {doc}"

        # Must contain exact 6-decimal values from authoritative JSON
        assert f"{val_auroc:.6f}" in text, (
            f"Expected validation AUROC {val_auroc:.6f} not found in {doc}"
        )
        assert f"{test_auroc:.6f}" in text, (
            f"Expected test AUROC {test_auroc:.6f} not found in {doc}"
        )
        assert f"{diff_auroc:.6f}" in text, (
            f"Expected delta AUROC {diff_auroc:.6f} not found in {doc}"
        )


def test_authoritative_knee_crop_pixel_spacing():
    """Verify authoritative knee-crop pixel spacing is 0.15 mm/pixel across code and documentation."""
    # 1. Verify code constant in imaging preprocessing if present
    prep_code = Path("src/imaging/full_preprocessing.py")
    if prep_code.is_file():
        text = prep_code.read_text(encoding="utf-8")
        assert "TARGET_SPACING_MM = 0.15" in text, (
            "TARGET_SPACING_MM in src/imaging/full_preprocessing.py must be 0.15"
        )

    # 2. Verify all public documentation asserting pixel spacing states 0.15 / 0.150 mm/pixel
    import re

    doc_paths = [
        Path("docs/imaging_pipeline.md"),
        Path("README.md"),
        Path("docs/portfolio_summary.md"),
        Path("docs/reproducibility.md"),
        Path("docs/repository_structure.md"),
    ]
    for doc in doc_paths:
        assert doc.is_file(), f"Missing public doc: {doc}"
        text = doc.read_text(encoding="utf-8")
        assert re.search(r"0\.150?\s*(\\text\{)?\s*mm(/|\s*per\s*)pixel", text) is not None, (
            f"Expected authoritative 0.15 mm/pixel specification not found in {doc}"
        )

        # Reject incorrect / stale claims
        assert "0.14 mm/pixel" not in text, f"Stale/incorrect 0.14 mm/pixel claim found in {doc}"
        assert "0.140 mm/pixel" not in text, f"Stale/incorrect 0.140 mm/pixel claim found in {doc}"


def test_public_release_license_contract():
    """Verify license file, allowlist inclusion, pyproject metadata, and README terms boundary."""
    # 1. Root LICENSE file exists and contains standard Apache 2.0 terms
    license_file = Path("LICENSE")
    assert license_file.is_file(), "Root LICENSE file missing"
    license_text = license_file.read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "http://www.apache.org/licenses/" in license_text

    # 2. Manifest allowlist inclusion
    manifest_path = Path("release/public_release_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "LICENSE" in manifest.get("allowlist", []), "LICENSE must be in release allowlist"

    # 3. pyproject.toml license metadata
    pyproject = Path("pyproject.toml")
    if pyproject.is_file():
        pyproj_text = pyproject.read_text(encoding="utf-8")
        assert "Apache-2.0" in pyproj_text, "pyproject.toml must specify Apache-2.0"

    # 4. README software license and OAI data terms boundary
    readme = Path("README.md")
    assert readme.is_file(), "README.md missing"
    readme_text = readme.read_text(encoding="utf-8")
    assert "Apache License 2.0" in readme_text, "README.md must state Apache License 2.0"
    assert "Osteoarthritis Initiative (OAI) source data" in readme_text
    assert "governed separately" in readme_text

    # 5. Verify no stale 'license pending' statements remain in public documentation
    stale_phrases = [
        "license pending",
        "license selection pending",
        "software license remains pending",
    ]
    for phrase in stale_phrases:
        assert phrase not in readme_text.lower(), f"Stale license phrase '{phrase}' in README.md"
