"""Authored synthetic output fixtures; no external executable or quantum table."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from hansenkit.xtb_features import (
    BOHR_ANGSTROM,
    DIPOLE_AU_TO_DEBYE,
    FEATURE_NAMES,
    FEATURE_UNITS,
    GAS_PROTOCOL,
    SCALAR_UNITS,
    VERSION,
    aggregate_xtb_conformers,
    parse_xtb_singlepoint,
)


@pytest.fixture
def output():
    # These arrays are arbitrary synthetic quantities, not a computed ethanol.
    numbers = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1])
    positions = np.arange(27, dtype=float).reshape(9, 3) / 10
    charges = np.zeros(9)
    charges[2], charges[8] = -0.4, 0.4
    atomic_dipoles = np.zeros((9, 3))
    atomic_dipoles[0] = [0.1, -0.2, 0.3]
    q_only = charges @ (positions / BOHR_ANGSTROM)
    full = q_only + atomic_dipoles.sum(0)
    # Preserve distinct SCC-result and final-wavefunction states.
    payload = {
        "program call": "D:/authored/xtb.exe geometry.xyz",
        "dipole / a.u.": (full + [1e-5, 0, 0]).tolist(),
        "partial charges": charges.tolist(),
        "atomic dipole moments": atomic_dipoles.tolist(),
        "number of electrons": 20,
        "number of unpaired electrons": 0,
        "HOMO-LUMO gap / eV": 4.0,
        "method": "GFN2-xTB",
        "xtb version": VERSION,
    }
    rows = "\n".join(
        f"{i + 1} {z} {'O' if z == 8 else 'C' if z == 6 else 'H'} 1.0 {q:.3f} 1.0 1.0"
        for i, (z, q) in enumerate(zip(numbers, charges, strict=True))
    )
    stdout = f""" : Hamiltonian GFN2-xTB :
 : restarted? false :
 : GBSA solvation false :
 : PC potential false :
 : max. iterations 250 :
 : electronic temp. 300.0000000 K :
 : accuracy 1.0000000 :
 : net charge 0 :
 : unpaired electrons 0 :
 *** convergence criteria satisfied after 9 iterations ***
 # Z covCN q C6AA α(0)
{rows}

 Mol. C6AA /au·bohr⁶ : 10.0
 Mol. C8AA /au·bohr⁸ : 20.0
 Mol. α(0) /au : 5.0
 q only: {q_only[0]:.3f} {q_only[1]:.3f} {q_only[2]:.3f}
 full: {full[0]:.3f} {full[1]:.3f} {full[2]:.3f} {np.linalg.norm(full) * DIPOLE_AU_TO_DEBYE:.3f}
"""
    review = {
        "audit_basis": "operator_assumption",
        "scope": "local_scientific_private_noncommercial",
        "execution_allowed": True,
        "redistribution_allowed": False,
        "qualified_hsp_backend": False,
        "protocol": copy.deepcopy(GAS_PROTOCOL),
    }
    return {
        "stdout": stdout,
        "stderr": "normal termination of xtb\n",
        "json_text": json.dumps(payload, indent=2),
        "smiles": "CCO",
        "atomic_numbers": numbers,
        "positions_angstrom": positions,
        "review": review,
        "review_sha256": "b" * 64,
    }


def test_complete_source_units_and_distinct_dipole_states(output):
    record = parse_xtb_singlepoint(**output)
    payload = json.loads(output["json_text"])
    np.testing.assert_allclose(record.full_dipole_au, payload["dipole / a.u."])
    assert record.mu_debye == pytest.approx(
        np.linalg.norm(record.full_dipole_au) * DIPOLE_AU_TO_DEBYE
    )
    assert record.gap_ev == 4
    assert record.alpha_bohr3 == 5
    assert record.c6_hartree_bohr6 == 10
    assert record.c8_hartree_bohr8 == 20
    assert record.diagnostics["scc_vs_final_wavefunction_dipole_max_abs_au"] == pytest.approx(1e-5)
    assert record.diagnostics["dipole_model"] == "scc_result_full_electronic_multipoles"
    assert record.donor_hydrogen_indices == (8,)
    assert record.acceptor_indices == (2,)
    restored_json = json.loads(json.dumps(record.to_dict()))
    assert restored_json["scalar_units"] == list(SCALAR_UNITS)
    assert restored_json["diagnostics"]["redistribution_allowed"] is False
    assert restored_json["diagnostics"]["qualified_hsp_backend"] is False
    assert restored_json["diagnostics"]["hsp_predictions"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "commercial"),
        ("redistribution_allowed", True),
        ("execution_allowed", False),
        ("qualified_hsp_backend", True),
        ("audit_basis", "assumed"),
    ],
)
def test_scoped_review_cannot_be_broadened(output, field, value):
    output["review"][field] = value
    with pytest.raises(ValueError, match="review SHA"):
        parse_xtb_singlepoint(**output)


def test_raw_file_digest_and_fixed_protocol(output):
    output["review_sha256"] = "not-a-raw-file-digest"
    with pytest.raises(ValueError, match="review SHA"):
        parse_xtb_singlepoint(**output)
    output["review_sha256"] = "a" * 64
    output["review"]["protocol"]["fallback"] = "UFF"
    with pytest.raises(ValueError, match="review SHA"):
        parse_xtb_singlepoint(**output)


@pytest.mark.parametrize(
    "change",
    [
        {"stderr": "abnormal termination of xtb"},
        {"returncode": 1},
        {"stderr": "normal termination of xtb\nERROR"},
    ],
)
def test_failed_process_never_yields_features(output, change):
    output.update(change)
    with pytest.raises(ValueError):
        parse_xtb_singlepoint(**output)


@pytest.mark.parametrize(
    "old,new",
    [
        ("convergence criteria satisfied", "not converged"),
        ("GBSA solvation false", "GBSA solvation true"),
        ("restarted? false", "restarted? true"),
        ("electronic temp. 300.0000000", "electronic temp. 1000.0000000"),
        ("max. iterations 250", "max. iterations 1000"),
        ("net charge 0", "net charge 1"),
    ],
)
def test_wrong_scientific_conditions_refused(output, old, new):
    output["stdout"] = output["stdout"].replace(old, new)
    with pytest.raises(ValueError):
        parse_xtb_singlepoint(**output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("method", "GFN1-xTB"),
        ("xtb version", "6.7.1"),
        ("number of electrons", 26),
        ("number of unpaired electrons", 1),
        ("HOMO-LUMO gap / eV", -1),
        ("HOMO-LUMO gap / eV", None),
        ("partial charges", [0.0] * 8),
        ("partial charges", [0.1] * 9),
        ("atomic dipole moments", [[0.0] * 3] * 8),
        ("dipole / a.u.", [0.0, 0.0]),
    ],
)
def test_invalid_json_quantities_and_valence_counts(output, key, value):
    payload = json.loads(output["json_text"])
    payload[key] = value
    output["json_text"] = json.dumps(payload)
    with pytest.raises(ValueError):
        parse_xtb_singlepoint(**output)


def test_bounded_optional_metadata_removal_never_repairs_science(output):
    raw = output["json_text"].replace("D:/authored/xtb.exe", r"D:\authored\xtb.exe")
    output["json_text"] = raw
    with pytest.raises(ValueError, match="Malformed xTB JSON"):
        parse_xtb_singlepoint(**output)
    record = parse_xtb_singlepoint(**output, repair_program_call=True)
    assert output["json_text"] == raw
    assert record.diagnostics["metadata_program_call_removed"] is True
    broken_science = raw.replace('"method": "GFN2-xTB"', r'"method": "bad\value"')
    output["json_text"] = broken_science
    with pytest.raises(ValueError):
        parse_xtb_singlepoint(**output, repair_program_call=True)


def test_duplicate_and_nonfinite_json_science_are_rejected(output):
    for raw in (
        output["json_text"].replace("4.0", "NaN"),
        output["json_text"].replace("4.0", "1e999"),
        output["json_text"].replace('"method":', '"method": "GFN2-xTB", "method":'),
    ):
        with pytest.raises(ValueError):
            parse_xtb_singlepoint(**{**output, "json_text": raw})


def test_wrong_atom_order_and_position_closure_rejected(output):
    output["atomic_numbers"] = output["atomic_numbers"][::-1]
    with pytest.raises(ValueError, match="atom order"):
        parse_xtb_singlepoint(**output)
    output["atomic_numbers"] = output["atomic_numbers"][::-1]
    output["positions_angstrom"][2] += 1.0
    with pytest.raises(ValueError, match="full/q-only"):
        parse_xtb_singlepoint(**output)


def test_population_atom_coverage_and_dispersion_units(output):
    for old, new in (
        ("3 8 O", "3 7 N"),
        ("Mol. C6AA /au·bohr⁶", "Mol. C6AA /kcal"),
        ("Mol. α(0) /au : 5.0", "Mol. α(0) /au : -5.0"),
    ):
        with pytest.raises(ValueError):
            parse_xtb_singlepoint(**{**output, "stdout": output["stdout"].replace(old, new)})


@pytest.mark.parametrize("smiles", ["CC.[Na+]", "[NH3+]CC[O-]", "[CH3]", "[13CH4]"])
def test_unsupported_source_graphs_cannot_be_qualified(output, smiles):
    output["smiles"] = smiles
    with pytest.raises(ValueError):
        parse_xtb_singlepoint(**output)


def test_complete_three_conformers_and_descriptive_units(output):
    record = parse_xtb_singlepoint(**output)
    features = aggregate_xtb_conformers([record] * 3, volumes_angstrom3=[10, 20, 40])
    assert features.names == FEATURE_NAMES and features.units == FEATURE_UNITS
    assert features.values.shape == (20,)
    np.testing.assert_allclose(features.values[1:10:2], 0, atol=1e-14)
    assert features.values[10] == pytest.approx(np.mean(5 / np.array([10, 20, 40])))
    assert features.values[11] == pytest.approx(np.mean(10 / np.array([10, 20, 40]) ** 2))
    assert features.values[17] == pytest.approx(0.4 / 3)
    assert features.values[18] == pytest.approx(-0.4 / 3)
    restored = json.loads(json.dumps(features.to_dict()))
    assert restored["diagnostics"]["equilibrium_qualified"] is False
    assert restored["diagnostics"]["conformer_statistics"] == "equal_weight_descriptive"
    assert restored["diagnostics"]["hsp_predictions"] is None


def test_partial_conformers_scope_forgery_and_bad_volumes_refused(output):
    record = parse_xtb_singlepoint(**output)
    with pytest.raises(ValueError, match="Exactly three"):
        aggregate_xtb_conformers([record] * 2, volumes_angstrom3=[10, 10])
    altered = copy.deepcopy(record.diagnostics)
    altered["qualified_hsp_backend"] = True
    with pytest.raises(ValueError, match="Conformer graph"):
        aggregate_xtb_conformers(
            [record, record, replace(record, diagnostics=altered)], volumes_angstrom3=[10, 10, 10]
        )
    for volume in ([0, 10, 10], [np.nan, 10, 10], [10, 10]):
        with pytest.raises(ValueError, match="geometric"):
            aggregate_xtb_conformers([record] * 3, volumes_angstrom3=volume)


def test_mutated_record_quantities_are_not_silently_clipped(output):
    record = parse_xtb_singlepoint(**output)
    record.scalars[0] = -1
    with pytest.raises(ValueError):
        aggregate_xtb_conformers([record] * 3, volumes_angstrom3=[10] * 3)
