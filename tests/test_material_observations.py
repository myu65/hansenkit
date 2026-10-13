import copy
import json
from pathlib import Path

import pytest

from hansenkit.cli import main
from hansenkit.material_observations import ReportedWeight, SourceObservationLookup
from hansenkit.provenance import file_hash


def test_literal_unknown_weight_roundtrip_without_unit_conversion(evidence):
    ledger, _, save, _, _, _, _ = evidence
    weight = {
        "kind": "Mw",
        "value_as_reported": "AUTHORED 12,345",
        "unit_as_reported": None,
        "method": None,
    }
    ledger["records"][0]["reported_molecular_weight"] = weight
    data, manifest = save()
    row = SourceObservationLookup(data, manifest).lookup({"material_id": "fixture-grade-Mw1000"})[
        "records"
    ][0]
    reported = row["reported_molecular_weight"]
    assert reported == {**weight, "value_g_mol": None}
    assert (
        ReportedWeight.model_validate_json(
            ReportedWeight.model_validate(reported).model_dump_json()
        ).model_dump()
        == reported
    )
    assert row["mn_g_mol"] is None
    assert row["hsp_components"] is None and row["hsp_predictions"] is None


def test_existing_normalized_weight_stays_qualified_without_literal():
    weight = ReportedWeight(kind="Mn", value_g_mol=1234, method="authored method")
    assert weight.value_g_mol == 1234
    assert weight.value_as_reported is None and weight.unit_as_reported is None


def test_literal_weight_unit_is_retained_without_automatic_normalization():
    weight = ReportedWeight(
        kind="unspecified",
        value_as_reported="authored nominal 4.5",
        unit_as_reported="source-specific unit",
        method=None,
    )
    assert weight.value_g_mol is None
    assert weight.unit_as_reported == "source-specific unit"


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"value_g_mol": None},
        {"value_as_reported": " "},
        {"value_as_reported": 1234},
        {"value_as_reported": "authored", "unit_as_reported": " "},
        {"value_g_mol": True},
        {"value_g_mol": 0},
        {"value_g_mol": float("inf")},
    ],
)
def test_weight_requires_positive_qualified_value_or_nonblank_source_literal(changes):
    with pytest.raises(ValueError):
        ReportedWeight.model_validate({"kind": "Mw", "method": None, **changes})


@pytest.fixture
def evidence(tmp_path):
    source = tmp_path / "authored-source.txt"
    source.write_text("Authored fixture, not a published measurement", encoding="utf8")
    preparation = tmp_path / "preparation.json"
    preparation.write_text('{"purpose":"authored preparation only"}', encoding="utf8")
    row = {
        "candidate_id": "fixture-0",
        "source_asset_sha256": file_hash(source),
        "source_member": "authored/member",
        "source_row_index_zero_based": 0,
        "material_id": "fixture-grade-Mw1000",
        "polymer_abbreviation": "AUTHORED",
        "reported_molecular_weight": {"kind": "Mw", "value_g_mol": 1000, "method": None},
        "mn_g_mol": None,
        "dispersity": None,
        "end_groups": None,
        "repeat_graph_with_source_supported_ports": None,
        "numeric_temperature_k": None,
        "material_phase": None,
        "source_reported_concentration_wv_percent": 5,
        "source_reported_polymer_mass_g": 0.25,
        "source_reported_solvent_volume_ml": 4.75,
        "observation_wait_h": 2,
        "solvent_name_as_reported": "Authored solvent",
        "solvent_identity_smiles": None,
        "binary_observation": 0,
        "qualitative_note_as_reported": "Authored partial outcome",
        "positive_definition": "Source includes colloidal positive observations",
        "negative_definition": "Source includes partial negative observations",
        "label_kind": "authored_observation_fixture",
        "allowed_role": "authored_fixture_only",
        "hsp_components": None,
        "hsp_predictions": None,
        "physical_accuracy_validated": False,
    }
    ledger = {
        "scope": "Authored observation fixtures only",
        "audit_basis": "documented_permission",
        "preparation_review_sha256": file_hash(preparation),
        "records": [row],
    }
    review = {
        "sha256": "0" * 64,
        "source_asset_path": source.name,
        "source_asset_sha256": file_hash(source),
        "preparation_review_path": preparation.name,
        "preparation_review_sha256": file_hash(preparation),
        "source_members": {
            "authored/member": {"row_count": 3, "material_ids": [row["material_id"]]}
        },
        "material_ids": [row["material_id"]],
        "source": "Original project-authored fixture",
        "source_license": "MIT",
        "permission_evidence": "Original fixture authored for these tests",
        "approved_by": "fixture author",
        "rights_status": "approved",
        "audit_basis": "documented_permission",
        "use_scope": "scientific_private_noncommercial",
        "allowed_role": "evaluation_only",
        "read_only_lookup_allowed": True,
        "training_allowed": False,
        "derived_labels_allowed": False,
        "derived_weights_allowed": False,
        "hsp_inference_allowed": False,
        "public_redistribution": False,
    }
    data, manifest = tmp_path / "ledger.json", tmp_path / "review.json"

    def save():
        data.write_text(json.dumps(ledger), encoding="utf8")
        review["sha256"] = file_hash(data)
        manifest.write_text(json.dumps(review), encoding="utf8")
        return data, manifest

    save()
    return ledger, review, save, data, manifest, source, preparation


@pytest.mark.parametrize(
    "key,value",
    [
        ("rights_status", "pending"),
        ("rights_status", "denied"),
        ("read_only_lookup_allowed", False),
        ("read_only_lookup_allowed", 1),
        ("training_allowed", True),
        ("hsp_inference_allowed", True),
        ("public_redistribution", True),
        ("permission_evidence", " "),
    ],
)
def test_permissions_before_source_or_record_access(evidence, monkeypatch, key, value):
    _, review, save, data, manifest, _, _ = evidence
    review[key] = value
    save()
    original = Path.read_bytes

    def only_review(path):
        assert path == manifest, "Source data was read before lookup permission"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", only_review)
    monkeypatch.setattr(
        "hansenkit.material_observations.file_hash", lambda _: pytest.fail("Opaque access")
    )
    with pytest.raises(ValueError):
        SourceObservationLookup(data, manifest)


def test_old_preparation_only_manifest_does_not_authorize(evidence):
    _, _, _, data, _, _, preparation = evidence
    with pytest.raises(ValueError):
        SourceObservationLookup(data, preparation)


@pytest.mark.parametrize("asset", ["source", "preparation", "ledger"])
def test_checksum_drift_before_candidate_decode(evidence, monkeypatch, asset):
    _, _, _, data, manifest, source, preparation = evidence
    path = {"source": source, "preparation": preparation, "ledger": data}[asset]
    path.write_text("Not even valid JSON; no decoding is allowed", encoding="utf8")
    import hansenkit.material_observations as module

    decode = module._decode
    calls = []

    def tracked(raw):
        calls.append(raw)
        assert len(calls) == 1, "Candidate data decoded before checksum gate"
        return decode(raw)

    monkeypatch.setattr(module, "_decode", tracked)
    with pytest.raises(ValueError, match="checksum"):
        SourceObservationLookup(data, manifest)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"source_asset_sha256": "0" * 64},
        {"source_member": "foreign/member"},
        {"source_row_index_zero_based": 3},
        {"material_id": "different-grade"},
        {"hsp_components": [1, 2, 3]},
        {"hsp_predictions": [1, 2, 3]},
        {"binary_observation": True},
        {"physical_accuracy_validated": True},
        {"label_kind": "experimental_compatibility_observation_candidate"},
    ],
)
def test_row_provenance_and_label_boundaries(evidence, changes):
    ledger, _, save, data, manifest, _, _ = evidence
    ledger["records"][0].update(changes)
    save()
    with pytest.raises(ValueError):
        SourceObservationLookup(data, manifest)


def test_distinct_reviewed_grade_cannot_move_between_source_members(evidence):
    ledger, review, save, data, manifest, _, _ = evidence
    review["material_ids"].append("other-grade")
    review["source_members"]["other/member"] = {"row_count": 1, "material_ids": ["other-grade"]}
    ledger["records"][0]["material_id"] = "other-grade"
    save()
    with pytest.raises(ValueError, match="support"):
        SourceObservationLookup(data, manifest)


def test_duplicate_source_ids_refuse_but_duplicate_query_keys_remain(evidence):
    ledger, _, save, data, manifest, _, _ = evidence
    duplicate = copy.deepcopy(ledger["records"][0])
    ledger["records"].append(duplicate)
    save()
    with pytest.raises(ValueError, match="distinct"):
        SourceObservationLookup(data, manifest)
    duplicate.update(candidate_id="fixture-1", source_row_index_zero_based=1)
    third = copy.deepcopy(duplicate)
    third.update(candidate_id="fixture-2", source_row_index_zero_based=2, binary_observation=1)
    ledger["records"].append(third)
    save()
    report = SourceObservationLookup(data, manifest).lookup({"material_id": "fixture-grade-Mw1000"})
    assert report["rows"] == 3
    assert report["same_query_key_outcome_disagreement"] is True
    assert [row["binary_observation"] for row in report["records"]] == [0, 0, 1]


def test_exact_query_conditions_and_provenance_round_trip(evidence):
    _, _, _, data, manifest, _, _ = evidence
    lookup = SourceObservationLookup(data, manifest)
    query = {"material_id": "fixture-grade-Mw1000", "solvent_name_as_reported": "Authored solvent"}
    report = json.loads(json.dumps(lookup.lookup(query)))
    assert report["status"] == "reported_source_observation" and report["rows"] == 1
    row = report["records"][0]
    assert row["reported_molecular_weight"]["kind"] == "Mw"
    assert row["reported_molecular_weight"]["value_g_mol"] == 1000
    for field in (
        "mn_g_mol",
        "end_groups",
        "numeric_temperature_k",
        "dispersity",
        "material_phase",
    ):
        assert row[field] is None
    assert row["source_reported_concentration_wv_percent"] == 5
    assert row["source_reported_polymer_mass_g"] == 0.25
    assert row["source_reported_solvent_volume_ml"] == 4.75
    assert row["qualitative_note_as_reported"] == "Authored partial outcome"
    assert report["data_sha256"] == file_hash(data) and report["review_sha256"] == file_hash(
        manifest
    )
    assert report["hsp_predictions"] is None and report["model_execution"] is False
    row["binary_observation"] = 1
    assert lookup.lookup(query)["records"][0]["binary_observation"] == 0
    assert lookup.lookup({"material_id": "fixture-grade-Mn1000"})["rows"] == 0
    assert (
        lookup.lookup(
            {"material_id": query["material_id"], "solvent_name_as_reported": "authored solvent"}
        )["rows"]
        == 0
    )
    for field, value in (("dp", 10), ("numeric_temperature_k", 298.15), ("smiles", "CCO")):
        with pytest.raises(ValueError):
            lookup.lookup({**query, field: value})


def test_duplicate_json_keys_are_rejected(evidence):
    _, _, _, data, manifest, _, _ = evidence
    raw = manifest.read_text(encoding="utf8")
    manifest.write_text(raw[:-1] + ', "read_only_lookup_allowed": true}', encoding="utf8")
    with pytest.raises(ValueError, match="Repeated JSON"):
        SourceObservationLookup(data, manifest)


def test_cli_writes_only_local_evidence_and_summary(evidence, tmp_path, capsys):
    _, _, _, data, manifest, _, _ = evidence
    output = tmp_path / "report.json"
    args = [
        "material-lookup",
        "--data",
        str(data),
        "--review",
        str(manifest),
        "--material-id",
        "fixture-grade-Mw1000",
        "--output",
        str(output),
    ]
    main(args)
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == 1 and "records" not in printed
    before = output.read_bytes()
    assert json.loads(before)["hsp_predictions"] is None
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and output.read_bytes() == before
