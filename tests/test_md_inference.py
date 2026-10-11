"""Original structures, owned frozen tensors and arbitrary labels; no MD executed."""

import copy
import hashlib
import json
import math
from importlib.metadata import version

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from hansenkit.chains import assemble_homopolymer
from hansenkit.cli import main
from hansenkit.descriptors import descriptor_provenance
from hansenkit.intensive import COUNT_NAMES, INTENSIVE_VERSION, intensive_features
from hansenkit.md_inference import MDCohesionPredictor, MDMaterialInput


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_features(unit="*CCO*", dp=10, tacticity="none"):
    first = assemble_homopolymer(unit, 1, "*C", "*C")
    second = assemble_homopolymer(unit, 2, "*C", "*C")
    mol = Chem.MolFromSmiles(first)
    mass = Descriptors.MolWt(mol) + (dp - 1) * (
        Descriptors.MolWt(Chem.MolFromSmiles(second)) - Descriptors.MolWt(mol)
    )
    intensive = intensive_features(first)
    n = mol.GetNumHeavyAtoms()
    chemical = np.r_[intensive.values, intensive.values[: len(COUNT_NAMES)] * n, n]
    return (
        first,
        mass,
        np.r_[chemical, math.log1p(dp), math.log1p(mass), 1 / dp, float(tacticity == "atactic")],
    )


def head(columns, intercept):
    return {
        "kind": "ridge",
        "mean": [0.0] * columns,
        "scale": [1.0] * columns,
        "coef": np.zeros((2, columns)).tolist(),
        "intercept": intercept,
    }


@pytest.fixture
def artifacts(tmp_path):
    capped, mass, x = source_features()
    encoder = {
        "encoder_id": "owned-test-vectors-no-MD-no-encoder",
        "fixed": True,
        "pretrained": False,
    }
    hashes = {
        k: hashlib.sha256(k.encode()).hexdigest()
        for k in ("plan", "fitting_review", "reservation", "selection", "source", "qualification")
    }
    calibration = {
        mode: {
            "nominal_coverage": 0.9,
            "target_scale": [0.2, 0.3],
            "family_score_count": 9,
            "order_statistic_1based": 9,
            "multiplier": 1.0,
            "interval_half_widths": [0.2, 0.3],
            "computed_reference_only": True,
        }
        for mode in ("A", "B", "C")
    }
    a = head(109, [2.0, 3.0])
    model = {
        "artifact_type": "computed_md_cohesion_regression",
        "label_kind": "computed_md_cohesion",
        "target_names": ["sp_vdw", "sp_ele"],
        "units": "MPa^0.5",
        "temperature_k": 300,
        "pressure_atm": 1,
        "forcefield": "GAFF2_mod",
        "source_sp_module_version": "0.1.1",
        "audit_basis": "operator_assumption",
        "use_scope": "scientific_private_noncommercial",
        "public_redistribution": False,
        "plan_sha256": hashes["plan"],
        "fitting_review_sha256": hashes["fitting_review"],
        "expanded_reservation_sha256": hashes["reservation"],
        "fresh_selection_sha256": hashes["selection"],
        "encoder_metadata": encoder,
        "models": {
            "A": a,
            "B": head(772, [4.0, 5.0]),
            "C": {"chemical": copy.deepcopy(a), "residual": head(772, [1.0, 1.0])},
        },
        "calibration": calibration,
        "training_feature_min": (x - 1).tolist(),
        "training_feature_max": (x + 1).tolist(),
        "training_only_weighted_mean": [2.0, 3.0],
        "hsp_predictions": None,
        "individual_polar_hydrogen_components_identified": False,
        "independent_experimental_accuracy_validated": False,
        "physical_hsp_backend_qualified": False,
    }
    metadata = {
        "encoder_metadata": encoder,
        "source_sha256": hashes["source"],
        "qualification_sha256": hashes["qualification"],
        "reservation_sha256": hashes["reservation"],
        "selection_sha256": hashes["selection"],
        "feature_version": INTENSIVE_VERSION,
        "descriptor_provenance": descriptor_provenance(),
    }
    support = {
        "source_uuid": np.array(["owned-fixture"]),
        "repeat_smiles": np.array(["*CCO*"]),
        "capped_smiles": np.array([capped]),
        "dp": np.array([10], dtype=int),
        "mn_g_mol": np.array([mass]),
        "tacticity": np.array(["none"]),
        "features": x[None],
        "embeddings": np.linspace(-0.1, 0.1, 768)[None].astype(np.float32),
        "metadata_json": np.array(json.dumps(metadata)),
    }
    review = {
        "plan_sha256": hashes["plan"],
        "fitting_review_sha256": hashes["fitting_review"],
        "reservation_sha256": hashes["reservation"],
        "fresh_selection_sha256": hashes["selection"],
        "source_sha256": hashes["source"],
        "qualification_sha256": hashes["qualification"],
        "runtime_versions": {p: version(p) for p in ("numpy", "scikit-learn", "rdkit")},
        "source": "Original owned tensor fixture; arbitrary labels; no MD executed",
        "declared_license": "Original fixture MIT",
        "permission_evidence": "Owned original synthetic fixture only",
        "approved_by": "test author",
        "audit_basis": "operator_assumption",
        "use_scope": "scientific_private_noncommercial",
        "rights_status": "approved",
        "local_cached_inference_allowed": True,
        "public_redistribution": False,
        "encoder_execution_allowed": False,
        "hsp_inference_allowed": False,
    }
    paths = {
        k: tmp_path / (k + (".npz" if k == "support" else ".json"))
        for k in ("model", "review", "support")
    }
    record = {
        "source_uuid": "owned-fixture",
        "repeat_unit_smiles": "*CCO*",
        "dp": 10,
        "mn_g_mol": mass,
        "dispersity": 1.0,
        "left_cap": "*C",
        "right_cap": "*C",
        "tacticity": "none",
        "architecture": "linear",
        "temperature_k": 300.0,
        "pressure_atm": 1.0,
        "forcefield": "GAFF2_mod",
        "sp_module_version": "0.1.1",
    }

    def write():
        paths["model"].write_text(json.dumps(model), encoding="utf-8")
        np.savez_compressed(paths["support"], **support)
        review["model_sha256"], review["support_sha256"] = (
            digest(paths["model"]),
            digest(paths["support"]),
        )
        paths["review"].write_text(json.dumps(review), encoding="utf-8")

    write()
    return model, support, review, paths, record, write


def predictor(artifacts):
    paths = artifacts[3]
    return MDCohesionPredictor(paths["model"], paths["review"], paths["support"])


@pytest.mark.parametrize("mode,expected", [("A", [2.0, 3.0]), ("B", [4.0, 5.0]), ("C", [3.0, 4.0])])
def test_original_owned_mode_roundtrip_and_provenance(artifacts, mode, expected):
    result = predictor(artifacts).predict(artifacts[4], mode)
    assert result["status"] == "computed_reference_prediction"
    assert list(result["predictions"].values()) == expected
    assert result["prediction_intervals"]["sp_vdw"]["lower"] == pytest.approx(expected[0] - 0.2)
    assert result["hsp_predictions"] is None
    assert not result["physical_hsp_backend_qualified"]
    assert not result["individual_polar_hydrogen_components_identified"]
    assert not result["encoder_execution_performed"]
    assert result["provenance"]["model_sha256"] == digest(artifacts[3]["model"])
    assert result["provenance"]["support_sha256"] == digest(artifacts[3]["support"])
    assert result == predictor(artifacts).predict(artifacts[4], mode)


@pytest.mark.parametrize(
    "field,value",
    [
        ("public_redistribution", True),
        ("encoder_execution_allowed", True),
        ("hsp_inference_allowed", True),
        ("local_cached_inference_allowed", False),
        ("local_cached_inference_allowed", 1),
        ("rights_status", "pending"),
        ("use_scope", "commercial"),
        ("permission_evidence", " "),
    ],
)
def test_inference_rights_fail_closed(artifacts, field, value):
    artifacts[2][field] = value
    artifacts[5]()
    with pytest.raises(ValueError):
        predictor(artifacts)


@pytest.mark.parametrize("which", ["model", "support"])
def test_checksum_mismatch_refused(artifacts, which):
    with artifacts[3][which].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        predictor(artifacts)


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_type", "HSPModel"),
        ("target_names", ["delta_d", "delta_p"]),
        ("units", "kcal/mol"),
        ("temperature_k", 298.15),
        ("pressure_atm", True),
        ("physical_hsp_backend_qualified", True),
        ("public_redistribution", 0),
        ("hsp_predictions", {}),
        ("plan_sha256", "0" * 64),
    ],
)
def test_model_contract_cannot_gain_qualification(artifacts, field, value):
    artifacts[0][field] = value
    artifacts[5]()
    with pytest.raises(ValueError):
        predictor(artifacts)


@pytest.mark.parametrize(
    "kind", ["shape", "scale", "nonfinite", "calibration", "bounds", "residual"]
)
def test_invalid_numeric_states_fail_before_prediction(artifacts, kind):
    model = artifacts[0]
    if kind == "shape":
        model["models"]["B"]["coef"] = [[0.0] * 771] * 2
    elif kind == "scale":
        model["models"]["A"]["scale"][0] = 0.0
    elif kind == "nonfinite":
        model["models"]["B"]["mean"][0] = float("nan")
    elif kind == "calibration":
        model["calibration"]["B"]["order_statistic_1based"] = 8
    elif kind == "bounds":
        model["training_feature_max"][0] = model["training_feature_min"][0] - 1
    else:
        model["models"]["C"]["chemical"]["intercept"][0] += 1
    artifacts[5]()
    with pytest.raises(ValueError):
        predictor(artifacts)


def test_runtime_and_encoder_metadata_are_exact(artifacts):
    artifacts[2]["runtime_versions"]["numpy"] = "unreviewed"
    artifacts[5]()
    with pytest.raises(ValueError, match="runtime"):
        predictor(artifacts)
    artifacts[2]["runtime_versions"]["numpy"] = version("numpy")
    meta = json.loads(artifacts[1]["metadata_json"].item())
    meta["encoder_metadata"]["encoder_id"] = "unreviewed-fallback"
    artifacts[1]["metadata_json"] = np.array(json.dumps(meta))
    artifacts[5]()
    with pytest.raises(ValueError, match="provenance"):
        predictor(artifacts)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("source_uuid", "unknown", "unregistered_source_material"),
        ("dp", 11, "registered_material_state_mismatch"),
        ("mn_g_mol", 10000.0, "registered_material_state_mismatch"),
        ("tacticity", "atactic", "registered_material_state_mismatch"),
        ("repeat_unit_smiles", "*COC*", "registered_material_state_mismatch"),
        ("left_cap", "*O", "unsupported_end_groups"),
        ("right_cap", "*[H]", "unsupported_end_groups"),
        ("dispersity", 1.1, "unsupported_molecular_weight_distribution"),
        ("temperature_k", 298.15, "unsupported_source_protocol"),
        ("pressure_atm", 2.0, "unsupported_source_protocol"),
        ("forcefield", "unreviewed", "unsupported_source_protocol"),
        ("sp_module_version", "new", "unsupported_source_protocol"),
        ("dp", True, "invalid_material_input"),
        ("dp", 10.0, "invalid_material_input"),
        ("repeat_unit_smiles", "[1*]CCO[2*]", "unsupported_repeat_chemistry"),
        ("repeat_unit_smiles", "*[13CH2]CO*", "unsupported_repeat_chemistry"),
        ("repeat_unit_smiles", "*C[N+](C)(C)CO*", "unsupported_repeat_chemistry"),
        ("repeat_unit_smiles", "*C[CH]CO*", "unsupported_repeat_chemistry"),
    ],
)
def test_material_refusals_retain_scope_and_blank_numbers(artifacts, field, value, reason):
    record = {**artifacts[4], field: value}
    result = predictor(artifacts).predict(record)
    assert result["status"] == "refused" and result["reasons"] == [reason]
    assert (
        result["predictions"] is result["prediction_intervals"] is result["hsp_predictions"] is None
    )
    assert result["provenance"]["model_sha256"] == digest(artifacts[3]["model"])


def test_mandatory_fields_and_extras_have_no_fallback(artifacts):
    p = predictor(artifacts)
    for field in MDMaterialInput.model_fields:
        record = artifacts[4].copy()
        del record[field]
        assert p.predict(record)["status"] == "refused"
    assert p.predict({**artifacts[4], "eo_po": {"sequence": "unknown"}})["status"] == "refused"
    assert p.predict(artifacts[4], "unknown")["status"] == "refused"


def test_source_assembly_and_features_rebuilt(artifacts):
    artifacts[1]["capped_smiles"] = np.array(["CCCC"])
    artifacts[5]()
    assert predictor(artifacts).predict(artifacts[4])["reasons"] == ["source_assembly_mismatch"]
    artifacts[1]["capped_smiles"] = np.array([source_features()[0]])
    artifacts[1]["features"][0, 0] += 0.1
    artifacts[5]()
    assert predictor(artifacts).predict(artifacts[4])["reasons"] == [
        "source_feature_reconstruction_mismatch"
    ]


def test_ood_and_negative_are_refused_without_clipping(artifacts):
    artifacts[0]["training_feature_max"][-4] = artifacts[1]["features"][0, -4] - 0.01
    artifacts[5]()
    assert predictor(artifacts).predict(artifacts[4])["reasons"] == [
        "outside_training_feature_bounds"
    ]
    artifacts[0]["training_feature_max"][-4] += 1
    artifacts[0]["models"]["B"]["intercept"][0] = -0.1
    artifacts[5]()
    result = predictor(artifacts).predict(artifacts[4], "B")
    assert result["reasons"] == ["negative_computed_prediction"] and result["predictions"] is None


def test_raw_computed_interval_is_not_clipped_or_physical(artifacts):
    artifacts[0]["calibration"]["A"]["multiplier"] = 20.0
    artifacts[0]["calibration"]["A"]["interval_half_widths"] = [4.0, 6.0]
    artifacts[5]()
    result = predictor(artifacts).predict(artifacts[4])
    assert result["prediction_intervals"]["sp_vdw"]["lower"] == -2.0
    assert result["interval_metadata"]["computed_reference_only"]
    assert not result["interval_metadata"]["physical_accuracy_or_exchangeability_qualified"]


def test_returned_provenance_cannot_mutate_predictor(artifacts):
    p = predictor(artifacts)
    result = p.predict(artifacts[4])
    result["provenance"]["source_encoder_metadata"]["encoder_id"] = "changed"
    assert (
        p.predict(artifacts[4])["provenance"]["source_encoder_metadata"]["encoder_id"]
        == artifacts[0]["encoder_metadata"]["encoder_id"]
    )


def test_no_encoder_execution_or_download(artifacts, monkeypatch):
    import hansenkit.encoders

    def forbidden(*args, **kwargs):
        raise AssertionError("No encoder may execute")

    monkeypatch.setattr(hansenkit.encoders, "get_encoder", forbidden)
    assert (
        predictor(artifacts).predict(artifacts[4], "B")["status"] == "computed_reference_prediction"
    )


def cli_args(artifacts, input_path, output_path, mode="A"):
    paths = artifacts[3]
    return [
        "md-cohesion-predict",
        "--model",
        str(paths["model"]),
        "--review",
        str(paths["review"]),
        "--support",
        str(paths["support"]),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--mode",
        mode,
    ]


@pytest.mark.parametrize("as_list", [False, True])
def test_cli_owned_fixture_mixed_report_and_no_private_stdout(artifacts, tmp_path, capsys, as_list):
    record = artifacts[4]
    content = [record, {**record, "dp": 11}] if as_list else record
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")
    main(cli_args(artifacts, input_path, output_path, "C"))
    report = json.loads(output_path.read_text())
    assert report["task"] == "computed_md_cohesion_cached_inference"
    assert report["reports"][0]["status"] == "computed_reference_prediction"
    if as_list:
        assert report["reports"][1]["status"] == "refused"
    assert report["hsp_predictions"] is None and not report["physical_hsp_backend_qualified"]
    assert record["source_uuid"] not in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(cli_args(artifacts, input_path, output_path))
    assert error.value.code == 2


def test_cli_invalid_artifact_creates_no_output(artifacts, tmp_path):
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(artifacts[4]), encoding="utf-8")
    artifacts[2]["model_sha256"] = "0" * 64
    artifacts[3]["review"].write_text(json.dumps(artifacts[2]), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(cli_args(artifacts, input_path, output_path))
    assert error.value.code == 2
    assert not output_path.exists()


@pytest.mark.parametrize("key", ["feature_version", "descriptor_provenance"])
def test_source_feature_provenance_must_be_explicit(artifacts, key):
    meta = json.loads(artifacts[1]["metadata_json"].item())
    del meta[key]
    artifacts[1]["metadata_json"] = np.array(json.dumps(meta))
    artifacts[5]()
    with pytest.raises(ValueError):
        predictor(artifacts)


@pytest.mark.parametrize(
    "case", ["scalar_uuid", "scalar_feature", "unhashable_audit", "encoder_not_fixed"]
)
def test_malformed_artifacts_raise_reviewable_value_error(artifacts, case):
    if case == "scalar_uuid":
        artifacts[1]["source_uuid"] = np.array("owned-fixture")
    elif case == "scalar_feature":
        artifacts[1]["features"] = np.array(1.0)
    elif case == "unhashable_audit":
        artifacts[0]["audit_basis"] = []
    else:
        artifacts[0]["encoder_metadata"]["fixed"] = False
    artifacts[5]()
    with pytest.raises(ValueError):
        predictor(artifacts)


@pytest.mark.parametrize("case", ["duplicate_uuid", "float_dp", "object_pickle"])
def test_support_identity_and_dtype_are_not_guessed(artifacts, case):
    support = artifacts[1]
    if case == "duplicate_uuid":
        for key, values in list(support.items()):
            if key != "metadata_json":
                support[key] = np.repeat(values, 2, axis=0)
    elif case == "float_dp":
        support["dp"] = support["dp"].astype(float)
    else:
        support["repeat_smiles"] = support["repeat_smiles"].astype(object)
    artifacts[5]()
    with pytest.raises(ValueError):
        predictor(artifacts)


def test_finite_state_overflow_is_refused_without_numbers(artifacts):
    model = artifacts[0]
    coefficients = np.zeros((2, 772))
    coefficients[:, -4:] = 1e308
    model["models"]["B"]["coef"] = coefficients.tolist()
    artifacts[5]()
    result = predictor(artifacts).predict(artifacts[4], "B")
    assert result["reasons"] == ["nonfinite_computed_prediction"]
    assert result["predictions"] is result["prediction_intervals"] is None
    assert result["hsp_predictions"] is None


def test_interval_overflow_is_refused_without_clipping(artifacts):
    model = artifacts[0]
    model["models"]["B"]["intercept"] = [1e308, 1e308]
    model["calibration"]["B"].update(
        target_scale=[1.0, 1.0], multiplier=1e308, interval_half_widths=[1e308, 1e308]
    )
    artifacts[5]()
    result = predictor(artifacts).predict(artifacts[4], "B")
    assert result["reasons"] == ["nonfinite_computed_interval"]
    assert result["predictions"] is result["prediction_intervals"] is None


def test_stereo_and_disconnected_inputs_do_not_match_a_source(artifacts):
    p = predictor(artifacts)
    for unit in ("*C[C@H](C)O*", "*CCO*.O"):
        result = p.predict({**artifacts[4], "repeat_unit_smiles": unit})
        assert result["status"] == "refused"
        assert result["predictions"] is None


def test_canonical_identity_alias_does_not_replace_port_chemistry(artifacts):
    p = predictor(artifacts)
    equivalent = p.predict({**artifacts[4], "repeat_unit_smiles": "*OCC*"})
    assert equivalent["status"] == "computed_reference_prediction"
    # The same periodic family can have a different cut/capped representation;
    # no grouping alias is used as a feature or authorization shortcut.
    different_cut = p.predict({**artifacts[4], "repeat_unit_smiles": "*COC*"})
    assert different_cut["reasons"] == ["registered_material_state_mismatch"]
