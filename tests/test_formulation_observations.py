"""Authored cells/conditions/categories only, never published P105 data."""

import copy
import json
from pathlib import Path

import pytest

from hansenkit import formulation_observations as lookup
from hansenkit.cli import main


def cell(row, column):
    return {
        "sheet_name": "Authored phase",
        "excel_row_one_based": row,
        "excel_column_one_based": column,
    }


def quantity(value, unit, qualifier="as_reported"):
    return {
        "value": value,
        "unit": unit,
        "unit_as_reported": unit,
        "qualifier": "unreported" if value is None else qualifier,
        "literal_as_reported": None if value is None else str(value),
        "source_cell": None,
    }


@pytest.fixture
def evidence(tmp_path):
    assets = {}
    for name in ("source_asset", "preparation_receipt", "classification_receipt"):
        path = tmp_path / (name + ".authored.txt")
        path.write_text("Owned fixture only: " + name, encoding="utf8")
        assets[name + "_path"] = path.name
        assets[name + "_sha256"] = lookup.sha(path)
    grade = {
        "source_grade_id": "AUTHORED-grade",
        "name_as_reported": "Authored nominal EO/PO grade; no external material",
        "nominal_eo_left": 7,
        "nominal_po": 11,
        "nominal_eo_right": 7,
        "block_counts_are_exact_DP": False,
        "reported_mass": {
            "value_g_mol": 1700,
            "statistic": "unspecified",
            "qualifier": "nominal",
            "literal_as_reported": "Authored nominal1700",
            "method_as_reported": None,
        },
        "mn_g_mol": None,
        "dispersity": None,
        "end_groups_as_reported": None,
        "source_supported_repeat_graph": None,
        "exact_material_structure_qualified": False,
    }
    conditions = {
        "polymer_wt_percent": quantity(2, "wt%"),
        "salt_name_as_reported": "Authored salt M",
        "salt_mol_l": quantity(0.2, "mol/L"),
        "silica_wt_percent": quantity(0.1, "wt%"),
        "silica_grade_as_reported": "Authored silica brand",
        "silica_charge_as_reported": None,
        "silica_protonation_as_reported": None,
        "temperature_c": quantity(22, "degC"),
        "pH": quantity(3, "pH", "approximately"),
        "pressure_pa": quantity(None, None),
        "observation_wait_h": quantity(None, None),
        "solvent_name_as_reported": None,
        "material_phase_as_reported": None,
        "polymer_present": True,
    }
    endpoints = {
        "turbidity-onset": {
            "name_as_reported": "Authored CloudPoint title, not a temperature label",
            "definition_as_reported": "Authored optical onset, not dissolved/undissolved",
            "literal_categories": ["Clear", "Turbid"],
            "source_outcome_cells": [cell(2, 3), cell(3, 3), cell(4, 3)],
        },
        "layer-growth": {
            "name_as_reported": "Authored layer growth",
            "definition_as_reported": "Authored layer on glass, not solubility",
            "literal_categories": ["NoGrowth", "Precip"],
            "source_outcome_cells": [cell(2, 4), cell(3, 4), cell(4, 4)],
        },
    }
    records = []
    for i, weight in enumerate((2, 2, 0)):
        for endpoint_id, category in (
            ("turbidity-onset", "Turbid" if i == 1 else "Clear"),
            ("layer-growth", "NoGrowth"),
        ):
            c = copy.deepcopy(conditions)
            c["polymer_wt_percent"] = quantity(weight, "wt%")
            c["polymer_present"] = weight > 0
            endpoint = endpoints[endpoint_id]
            records.append(
                {
                    "candidate_id": f"authored-{i}-{endpoint_id}",
                    "formulation_id": f"authored-formulation-{i}",
                    "source_asset_sha256": assets["source_asset_sha256"],
                    "source_outcome_cell": endpoint["source_outcome_cells"][i],
                    "source_title_cell": cell(1, 3 if endpoint_id == "turbidity-onset" else 4),
                    "source_axis_cells": [cell(i + 2, 1), cell(1, 2)],
                    "source_xf_index": 0,
                    "source_formula_origin": "unknown",
                    "source_grade_id": grade["source_grade_id"],
                    "conditions": c,
                    "endpoint_id": endpoint_id,
                    "endpoint_name_as_reported": endpoint["name_as_reported"],
                    "definition_as_reported": endpoint["definition_as_reported"],
                    "literal_category_as_reported": category,
                    "numerical_endpoint_value": None,
                    "interpreted_binary_solubility": None,
                    "label_kind": "authored_formulation_fixture",
                    "allowed_role": "authored_fixture_only",
                    "hsp_components": None,
                    "hsp_predictions": None,
                    "physical_accuracy_validated": False,
                }
            )
    ledger = {
        "scope": "Owned phase/formulation fixtures only",
        "audit_basis": "documented_permission",
        "use_scope": "scientific_private_noncommercial",
        "allowed_role": "authored_fixture_only",
        **{k: assets[k] for k in assets if k.endswith("_sha256")},
        "source_grades": {grade["source_grade_id"]: grade},
        "source_metadata_claims": [
            {
                "literal_as_reported": "Authored metadata claim; solution charge unknown",
                "source_pointer": "authored-metadata:/method",
                "interpretation": "unqualified_source_metadata_claim",
            }
        ],
        "records": records,
    }
    review = {
        "ledger_sha256": "0" * 64,
        **assets,
        "source_sheets": {"Authored phase": {"row_count": 5, "column_count": 5}},
        "source_grades": copy.deepcopy(ledger["source_grades"]),
        "endpoints": endpoints,
        "expected_outcome_records": 6,
        "expected_formulations": 3,
        "expected_polymer_present_formulations": 2,
        "expected_zero_polymer_controls": 1,
        "expected_unknown_polymer_presence": 0,
        "source": "Own arbitrary formulation fixtures, not published measurements",
        "declared_license": "MIT own fixture, no external data",
        "permission_evidence": "Own authored fixture only",
        "approved_by": "fixture author",
        "rights_status": "approved",
        "audit_basis": "documented_permission",
        "use_scope": "scientific_private_noncommercial",
        "allowed_role": "authored_fixture_only",
        "read_only_lookup_allowed": True,
        "training_allowed": False,
        "derived_labels_allowed": False,
        "derived_weights_allowed": False,
        "hsp_inference_allowed": False,
        "physical_accuracy_validated": False,
        "public_redistribution": False,
    }
    data, manifest = tmp_path / "ledger.authored.json", tmp_path / "lookup-review.authored.json"

    def save():
        data.write_text(json.dumps(ledger, allow_nan=False), encoding="utf8")
        review["ledger_sha256"] = lookup.sha(data)
        manifest.write_text(json.dumps(review, allow_nan=False), encoding="utf8")

    save()
    return ledger, review, save, data, manifest


def test_pair_endpoints_repeats_conflicts_controls_unknown_grade_and_no_predictions(evidence):
    ledger, _, _, data, review = evidence
    reader = lookup.FormulationObservationLookup(data, review)
    result = reader.lookup({"source_grade_id": "AUTHORED-grade"})
    assert result["outcome_records"] == 6 and result["formulations"] == 3
    assert result["polymer_present_formulations"] == 2
    assert result["zero_polymer_control_formulations"] == 1
    assert result["conflicting_literal_outcomes_at_same_reported_conditions"]
    assert [r["literal_category_as_reported"] for r in result["records"]] == [
        "Clear",
        "NoGrowth",
        "Turbid",
        "NoGrowth",
        "Clear",
        "NoGrowth",
    ]
    assert result["source_grade"]["reported_mass"]["statistic"] == "unspecified"
    assert (
        result["source_grade"]["mn_g_mol"]
        is result["source_grade"]["end_groups_as_reported"]
        is None
    )
    assert result["hsp_predictions"] is None and result["model_execution"] is False
    assert result["training_allowed"] is result["derived_labels_allowed"] is False
    assert result["allowed_role"] == "authored_fixture_only"
    assert {r["allowed_role"] for r in result["records"]} == {"authored_fixture_only"}
    assert "polymer_wt_percent" in result["records"][0]["conditions"]
    assert "p105_wt_percent" not in result["records"][0]["conditions"]
    assert result["records"][0]["conditions"]["silica_charge_as_reported"] is None
    assert result["records"][0]["conditions"]["silica_protonation_as_reported"] is None
    assert (
        result["source_metadata_claims"][0]["interpretation"] == "unqualified_source_metadata_claim"
    )
    result["records"][0]["literal_category_as_reported"] = "changed"
    assert (
        reader.lookup({"source_grade_id": "AUTHORED-grade"})["records"][0] == ledger["records"][0]
    )
    assert reader.lookup({"source_grade_id": "AUTHORED-new-grade"})["outcome_records"] == 0
    with pytest.raises(ValueError):
        reader.lookup(
            {"source_grade_id": "AUTHORED-grade", "conditions": {"p105_wt_percent": None}}
        )


def test_exact_values_units_qualifiers_and_unknown_filters_no_nearest(evidence):
    _, _, _, data, review = evidence
    reader = lookup.FormulationObservationLookup(data, review)
    query = {
        "source_grade_id": "AUTHORED-grade",
        "conditions": {
            "polymer_wt_percent": {"value": 2, "unit": "wt%", "qualifier": "as_reported"},
            "pH": {"value": 3, "unit": "pH", "qualifier": "approximately"},
            "polymer_present": True,
        },
    }
    result = json.loads(json.dumps(reader.lookup(query)))
    assert result["outcome_records"] == 4
    assert result["records"][0]["conditions"]["pressure_pa"]["value"] is None
    for field, changed in (("value", 2.01), ("unit", "w/v%")):
        q = copy.deepcopy(query)
        q["conditions"]["polymer_wt_percent"][field] = changed
        assert reader.lookup(q)["outcome_records"] == 0
    q = copy.deepcopy(query)
    q["conditions"]["pH"]["qualifier"] = "as_reported"
    assert reader.lookup(q)["outcome_records"] == 0
    assert (
        reader.lookup({"source_grade_id": "AUTHORED-grade", "conditions": {"pressure_pa": None}})[
            "outcome_records"
        ]
        == 6
    )
    assert (
        reader.lookup(
            {
                "source_grade_id": "AUTHORED-grade",
                "conditions": {
                    "pressure_pa": {
                        "value": 101325,
                        "unit": "Pa",
                        "qualifier": "as_reported",
                    }
                },
            }
        )["outcome_records"]
        == 0
    )
    zero = reader.lookup(
        {"source_grade_id": "AUTHORED-grade", "conditions": {"polymer_present": False}}
    )
    assert zero["outcome_records"] == 2 and zero["polymer_present_formulations"] == 0


def test_grouping_numeric_zero_and_integer_float_semantics_preserve_all_source_literals(evidence):
    ledger, review, save, data, manifest = evidence
    for index, record in enumerate(ledger["records"]):
        value = (0, 0.0, -0.0)[index // 2]
        record["conditions"]["polymer_wt_percent"] = quantity(value, "wt%")
        record["conditions"]["polymer_present"] = False
        record["conditions"]["temperature_c"] = quantity(22 if index % 2 == 0 else 22.0, "degC")
    review["expected_polymer_present_formulations"] = 0
    review["expected_zero_polymer_controls"] = 3
    save()
    report = lookup.FormulationObservationLookup(data, manifest).lookup(
        {
            "source_grade_id": "AUTHORED-grade",
            "conditions": {
                "polymer_wt_percent": {"value": 0, "unit": "wt%", "qualifier": "as_reported"},
                "temperature_c": {"value": 22.0, "unit": "degC", "qualifier": "as_reported"},
            },
        }
    )
    assert report["outcome_records"] == 6 and report["zero_polymer_control_formulations"] == 3
    assert report["conflicting_literal_outcomes_at_same_reported_conditions"] is True
    literals = [
        r["conditions"]["polymer_wt_percent"]["literal_as_reported"] for r in report["records"]
    ]
    assert literals == ["0", "0", "0.0", "0.0", "-0.0", "-0.0"]
    assert isinstance(report["records"][0]["conditions"]["temperature_c"]["value"], int)
    assert isinstance(report["records"][1]["conditions"]["temperature_c"]["value"], float)


@pytest.mark.parametrize(
    "field,value",
    [
        ("read_only_lookup_allowed", False),
        ("read_only_lookup_allowed", 1),
        ("training_allowed", True),
        ("derived_labels_allowed", True),
        ("hsp_inference_allowed", True),
        ("public_redistribution", True),
        ("rights_status", "pending"),
    ],
)
def test_rights_before_source_access_or_candidate_decode(evidence, monkeypatch, field, value):
    _, review, save, data, manifest = evidence
    review[field] = value
    save()
    original = Path.read_bytes

    def only_review(path):
        assert path == manifest, "No upstream/data access before exact lookup rights"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", only_review)
    monkeypatch.setattr(lookup, "sha", lambda _: pytest.fail("No opaque upstream read"))
    with pytest.raises(ValueError):
        lookup.FormulationObservationLookup(data, manifest)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda ledger, review: ledger["records"][0].update(source_outcome_cell=cell(2, 4)),
        lambda ledger, review: ledger["records"][0].update(source_outcome_cell=cell(2, 5)),
        lambda ledger, review: ledger["records"][0].update(literal_category_as_reported="soluble"),
        lambda ledger, review: ledger["records"][0].update(interpreted_binary_solubility=1),
        lambda ledger, review: ledger["records"][0]["conditions"].update(polymer_present=False),
        lambda ledger, review: ledger["records"][0]["conditions"]["polymer_wt_percent"].update(
            unit="w/v%"
        ),
        lambda ledger, review: ledger["records"].pop(),
        lambda ledger, review: ledger["records"][1]["conditions"]["salt_mol_l"].update(value=0.3),
        lambda ledger, review: ledger["source_grades"]["AUTHORED-grade"].update(mn_g_mol=1700),
    ],
)
def test_cell_endpoint_units_roles_population_and_pair_coherence(evidence, mutator):
    ledger, review, save, data, manifest = evidence
    mutator(ledger, review)
    save()
    with pytest.raises(ValueError):
        lookup.FormulationObservationLookup(data, manifest)


def test_upstream_checksum_before_ledger_decode_and_duplicateJSON(evidence, monkeypatch):
    _, review, _, data, manifest = evidence
    original = lookup.decode
    calls = []

    def counted(raw):
        calls.append(raw)
        assert len(calls) == 1, "No ledger decode before upstream hashes"
        return original(raw)

    monkeypatch.setattr(lookup, "decode", counted)
    (manifest.parent / review["classification_receipt_path"]).write_bytes(b"drift")
    with pytest.raises(ValueError, match="checksum"):
        lookup.FormulationObservationLookup(data, manifest)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Repeated JSON"):
        original(b'{"training_allowed":false,"training_allowed":true}')


def test_disabled_review_template_refuses_before_any_source_access(monkeypatch):
    template = Path(__file__).parents[1] / "examples/formulation-review.template.json"
    monkeypatch.setattr(lookup, "sha", lambda _: pytest.fail("Disabled template read an asset"))
    with pytest.raises(ValueError):
        lookup.FormulationObservationLookup("UNOPENED_SOURCE_LEDGER", template)


def test_review_ledger_role_mismatch_cannot_promote_authored_fixture_before_query(evidence):
    _, review, save, data, manifest = evidence
    review["allowed_role"] = "evaluation_only"
    save()
    with pytest.raises(ValueError, match="role provenance"):
        lookup.FormulationObservationLookup(data, manifest)


def test_cli_exclusive_json_roundtrip_summary_only(evidence, tmp_path, capsys):
    _, _, _, data, manifest = evidence
    query, output = tmp_path / "query.json", tmp_path / "reported.json"
    query.write_text(
        json.dumps({"source_grade_id": "AUTHORED-grade", "endpoint_id": "layer-growth"}),
        encoding="utf8",
    )
    argv = [
        "formulation-lookup",
        "--data",
        str(data),
        "--review",
        str(manifest),
        "--query",
        str(query),
        "--output",
        str(output),
    ]
    main(argv)
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "status": "reported_formulation_observations",
        "outcome_records": 3,
        "formulations": 3,
    }
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["records"][0]["literal_category_as_reported"] == "NoGrowth"
    assert result["records"][0]["source_outcome_cell"] == cell(2, 4)
    assert result["ledger_sha256"] == lookup.sha(data)
    before = output.read_bytes()
    with pytest.raises(SystemExit) as failed:
        main(argv)
    assert failed.value.code == 2
    assert output.read_bytes() == before
