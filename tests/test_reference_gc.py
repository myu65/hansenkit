import json

import numpy as np
import pytest

from hansenkit.chains import assemble_homopolymer
from hansenkit.cli import main
from hansenkit.provenance import file_hash
from hansenkit.reference_gc import COEFFICIENT_UNITS, GROUPS, ReferenceGC, source_group_counts


@pytest.fixture
def reference(tmp_path):
    # Original arbitrary arithmetic constants; no physical coefficient table.
    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "units": COEFFICIENT_UNITS,
                "polar_convention": "rss_of_group_type_force_sums",
                "reference_temperature_k": None,
                "groups": {g: {"fd": 2, "fp": 3, "eh": 4, "volume": 1} for g in GROUPS},
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "sha256": file_hash(parameters),
                "source": "Original mathematical test fixture",
                "declared_license": "MIT authored fixture; arbitrary numbers",
                "permission_evidence": "Authored for the project, no physical validity",
                "approved_by": "test author",
                "audit_basis": "documented_permission",
                "rights_status": "approved",
                "local_reference_calculation_allowed": True,
            }
        ),
        encoding="utf-8",
    )
    return parameters, review


def test_typed_atom_ownership_and_reference_units(reference):
    parameters, review = reference
    assert source_group_counts("CC(=O)OCCO") == {
        "ester": 1,
        "alcohol": 1,
        "methyl": 1,
        "methylene": 2,
    }
    result = ReferenceGC(parameters, review).molecule("CCO")
    np.testing.assert_allclose(
        [result[k] for k in ["delta_d", "delta_p", "delta_h"]], [2, np.sqrt(3), 2]
    )
    assert result["units"] == "MPa^0.5" and result["group_coverage"] == 1
    assert result["reference_temperature_k"] is None
    assert not result["physical_accuracy_validated"]
    assert not result["training_or_pseudo_label_generation_allowed"]


@pytest.mark.parametrize(
    "smiles", ["CC(=O)O", "CCOCC", "C1CCC1", "[Na+].[Cl-]", "[13CH3]CO", "*CC", "C", "CC(C)(C)O"]
)
def test_source_unsupported_groups_never_produce_fallback(smiles, reference):
    with pytest.raises(ValueError):
        ReferenceGC(*reference).molecule(smiles)


def test_local_rights_and_checksum_required(reference):
    parameters, review = reference
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["local_reference_calculation_allowed"] = False
    review.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="permission"):
        ReferenceGC(parameters, review)
    payload["local_reference_calculation_allowed"] = True
    payload["sha256"] = "0" * 64
    review.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ReferenceGC(parameters, review)


def test_unknown_units_and_nonpositive_total_volume_refused(reference):
    parameters, review = reference
    body = json.loads(parameters.read_text(encoding="utf-8"))
    body["units"]["volume"] = "m^3/mol"
    parameters.write_text(json.dumps(body), encoding="utf-8")
    authorization = json.loads(review.read_text(encoding="utf-8"))
    authorization["sha256"] = file_hash(parameters)
    review.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="units"):
        ReferenceGC(parameters, review)
    body["units"] = COEFFICIENT_UNITS
    for coefficient in body["groups"].values():
        coefficient["volume"] = 0
    parameters.write_text(json.dumps(body), encoding="utf-8")
    authorization["sha256"] = file_hash(parameters)
    review.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="volume"):
        ReferenceGC(parameters, review).molecule("CCO")


def test_bounded_direct_and_large_repeat_reference_agreement(reference):
    calculator = ReferenceGC(*reference)
    repeat = "[1*]CC(C)[2*]"
    direct = calculator.molecule(assemble_homopolymer(repeat, 128))
    bounded = calculator.homopolymer(repeat, 128, left_cap=None, right_cap=None)
    np.testing.assert_allclose(
        [direct[k] for k in ["delta_d", "delta_p", "delta_h"]],
        [bounded[k] for k in ["delta_d", "delta_p", "delta_h"]],
        rtol=0,
        atol=1e-12,
    )
    huge = calculator.homopolymer(repeat, 10**9, left_cap=None, right_cap=None)
    assert huge["bounded_additivity_checks"] == [4, 16, 64]
    assert all(np.isfinite(huge[k]) for k in ["delta_d", "delta_p", "delta_h"])
    with pytest.raises(TypeError):
        calculator.homopolymer(repeat, 100)
    with pytest.raises(ValueError, match="repeat count"):
        calculator.homopolymer(repeat, 1.5, left_cap=None, right_cap=None)


def test_explicit_size_refused_before_native_canonical_serialization():
    with pytest.raises(ValueError, match="heavy-atom budget"):
        source_group_counts("CC" * 257)


def test_reference_cli_requires_explicit_chain_caps(reference, capsys):
    parameters, review = reference
    options = ["reference-gc", "--parameters", str(parameters), "--review", str(review)]
    main(options + ["--smiles", "CCO"])
    assert json.loads(capsys.readouterr().out)["value_kind"] == "coefficient_reference_hsp"
    with pytest.raises(SystemExit) as error:
        main(options + ["--repeat-unit", "[1*]CC[2*]", "--repeat-count", "1000000000"])
    assert error.value.code == 2
    assert "both explicit caps" in capsys.readouterr().err
    main(
        options
        + [
            "--repeat-unit",
            "[1*]CC[2*]",
            "--repeat-count",
            "1000000000",
            "--left-cap",
            "hydrogen",
            "--right-cap",
            "hydrogen",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["repeat_count"] == 10**9 and not result["physical_accuracy_validated"]
    with pytest.raises(SystemExit):
        main(options + ["--smiles", "CCO", "--left-cap", "hydrogen"])
