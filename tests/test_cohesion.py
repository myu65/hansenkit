import csv
import json

import pytest

from hansenkit.cli import main
from hansenkit.cohesion import MD_UNITS, MDCohesion, prepare_md_cohesion
from hansenkit.data import load_dataset
from hansenkit.provenance import DatasetManifest, file_hash


@pytest.fixture
def md_files(tmp_path):
    data = tmp_path / "md.csv"
    common = {
        "temp": 300,
        "press": 1,
        "check_eq": "True",
        "smiles_list": "*CC*",
        "smiles_ter_1": "*C",
        "smiles_ter_2": "",
        "forcefield": "authored-math-fixture",
        "RadonPy_ver": "authored-test",
        "preset_sp_ver": "authored-test",
        "sp_ced": 100,
        "sp_total": 9,
        "sp_vdw": 6,
        "sp_ele": 5,
    }
    rows = [
        {"UUID": "valid", **common},
        {"UUID": "missing", **common, "sp_ele": ""},
        {"UUID": "no-equilibrium", **common, "check_eq": "False"},
        {"UUID": "bad-conditions", **common, "temp": 0},
    ]
    with data.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "sha256": file_hash(data),
                "source": "Original mathematical MD table fixture",
                "declared_license": "MIT authored fixture",
                "permission_evidence": "Authored here",
                "approved_by": "test author",
                "audit_basis": "documented_permission",
                "rights_status": "approved",
                "local_preparation_allowed": True,
                "source_units": MD_UNITS,
                "definition_evidence": "Fixture defines separately averaged values",
            }
        ),
        encoding="utf-8",
    )
    return data, review, tmp_path / "prepared"


def test_md_preparation_preserves_unknown_ends_and_refuses_missing_values(md_files):
    data, review, out = md_files
    report = prepare_md_cohesion(data, review, out)
    assert report["counts"] == {
        "source_rows": 4,
        "prepared_rows": 1,
        "missing_nonfinite_negative_observable_or_invalid_conditions": 2,
        "equilibrium_not_declared": 1,
    }
    observation = json.loads((out / "observations.jsonl").read_text(encoding="utf-8"))
    assert observation["material_metadata"]["smiles_ter_2"] is None
    assert observation["observation"]["hsp_predictions"] is None
    assert not report["training_allowed"] and report["hsp_three_component_targets"] == 0
    assert not report["holdout_family_isolation_qualified"]
    assert report["prepared_sha256"] == file_hash(out / "observations.jsonl")
    with pytest.raises(ValueError, match="already exists"):
        prepare_md_cohesion(data, review, out)


def test_mean_square_root_values_are_not_forced_into_exact_energy_identity():
    observation = MDCohesion(
        temperature_k=300, pressure_atm=1, sp_ced=100, sp_total=9, sp_vdw=6, sp_ele=5
    )
    assert observation.report()["mean_energy_diagnostic_flags"] == []
    assert 9**2 != 100
    assert observation.report()["individual_polar_hydrogen_components_identified"] is False
    with pytest.raises(ValueError):
        MDCohesion(**observation.model_dump(), delta_p=3, delta_h=4)
    for value in [-1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            MDCohesion(**{**observation.model_dump(), "sp_ele": value})


def test_md_source_units_permission_and_checksum_required(md_files):
    data, review, out = md_files
    original = json.loads(review.read_text(encoding="utf-8"))
    for update in [
        {"local_preparation_allowed": False},
        {"sha256": "0" * 64},
        {"source_units": {**MD_UNITS, "sp_ced": "kcal/mol"}},
        {"training_allowed": True},
    ]:
        review.write_text(json.dumps({**original, **update}), encoding="utf-8")
        with pytest.raises(ValueError):
            prepare_md_cohesion(data, review, out)
        assert not out.exists()


def test_md_cli_and_missing_column_refusal(md_files, capsys):
    data, review, out = md_files
    main(["md-cohesion-prepare", "--data", str(data), "--review", str(review), "--out", str(out)])
    assert json.loads(capsys.readouterr().out)["hsp_three_component_targets"] == 0
    data.write_text("UUID,temp\nexample,300\n", encoding="utf-8")
    body = json.loads(review.read_text(encoding="utf-8"))
    body["sha256"] = file_hash(data)
    review.write_text(json.dumps(body), encoding="utf-8")
    other = out.parent / "missing-column-output"
    with pytest.raises(ValueError, match="lacks MD"):
        prepare_md_cohesion(data, review, other)
    assert not other.exists()


def test_md_observables_cannot_silently_be_read_as_hsp_training_labels(md_files):
    data, _, out = md_files
    body = {
        "dataset_id": "authored-fixture",
        "sha256": file_hash(data),
        "label_kind": "synthetic",
        "source": "Original authored test fixture",
        "license": "MIT",
        "rights_status": "approved",
        "permission_evidence": "Authored here",
        "approved_by": "test author",
        "training_allowed": True,
        "derived_weights_allowed": True,
    }
    manifest = out.parent / "hsp-manifest.json"
    manifest.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="required columns"):
        load_dataset(data, manifest)
    with pytest.raises(ValueError):
        DatasetManifest(**{**body, "label_kind": "computed_md_cohesion"})
