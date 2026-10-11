import copy

import numpy as np
import pytest
from rdkit.Chem import AllChem, rdForceFieldHelpers

from hansenkit.physics_features import (
    CHARGE_NAMES,
    CONFORMER_PROTOCOL,
    FEATURE_NAMES,
    GEOMETRY_NAMES,
    TOPOLOGY_NAMES,
    runtime_assets,
    signed_dipole_norm,
    structural_proxies,
)


@pytest.fixture(scope="module")
def geometry_review():
    return {
        "assets": runtime_assets(),
        "protocol": copy.deepcopy(CONFORMER_PROTOCOL),
        "audit_basis": "operator_assumption",
        "scope": "local_scientific_private_noncommercial",
        "geometry_execution_allowed": True,
        "redistribution_allowed": False,
    }


def test_charge_sites_and_no_implicit_3d():
    ethanol = structural_proxies("CCO")
    assert ethanol.names == FEATURE_NAMES
    np.testing.assert_allclose(ethanol.values[:2], [1 / 3, 1 / 3])
    assert ethanol.diagnostics["charge_status"] == "complete"
    assert abs(sum(ethanol.diagnostics["gasteiger_partial_charges"])) < 1e-5
    assert np.isnan(ethanol.values[-len(GEOMETRY_NAMES) :]).all()
    assert ethanol.diagnostics["geometry"]["status"] == "not_requested"
    assert not ethanol.diagnostics["hydrogen_bond_strength_measured"]
    amide = structural_proxies("CC(N)=O")
    assert len(amide.diagnostics["donor_atoms"]) == 1
    assert len(amide.diagnostics["acceptor_atoms"]) == 1


@pytest.mark.parametrize(
    "smiles", ["C[N+](C)(C)C", "[NH3+]CC(=O)[O-]", "CCO.O", "[CH3]", "*CC", "[13CH4]"]
)
def test_neutrality_does_not_rewrite_or_qualify(smiles):
    with pytest.raises(ValueError, match="neutral"):
        structural_proxies(smiles)


def test_nitro_classification_keeps_original_charged_graph():
    output = structural_proxies("C[N+](=O)[O-]")
    assert "[N+]" in output.canonical_smiles
    assert "[O-]" in output.canonical_smiles
    assert output.diagnostics["qualified_hsp_backend"] is False


def test_parameter_failure_never_substitutes_zero(monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["throwOnParamFailure"] is True
        raise ValueError("Unavailable parameters")

    monkeypatch.setattr(AllChem, "ComputeGasteigerCharges", fail)
    output = structural_proxies("CCO")
    start = len(TOPOLOGY_NAMES)
    assert np.isnan(output.values[start : start + len(CHARGE_NAMES)]).all()
    assert output.diagnostics["charge_status"] == "missing_or_invalid_parameters"
    assert np.isfinite(output.values[:start]).all()


def test_actual_local_geometry_repeatability_and_full_trace(geometry_review):
    first = structural_proxies("CCO", geometry_review=geometry_review)
    second = structural_proxies("OCC", geometry_review=geometry_review)
    assert first.diagnostics["geometry"]["status"] == "complete_fixed_protocol"
    assert len(first.diagnostics["geometry"]["coordinates_angstrom"]) == 3
    assert all(o["status"] == 0 for o in first.diagnostics["geometry"]["optimization"])
    np.testing.assert_array_equal(first.values, second.values)
    assert np.isfinite(first.values).all()


def test_any_unconverged_conformer_invalidates_entire_block(geometry_review, monkeypatch):
    monkeypatch.setattr(
        rdForceFieldHelpers,
        "MMFFOptimizeMoleculeConfs",
        lambda *args, **kwargs: [(0, -2.0), (1, -2.1), (0, -2.2)],
    )
    output = structural_proxies("CCO", geometry_review=geometry_review)
    assert output.diagnostics["geometry"]["status"] == "incomplete_optimization"
    assert len(output.diagnostics["geometry"]["coordinates_angstrom"]) == 3
    assert np.isnan(output.values[-len(GEOMETRY_NAMES) :]).all()


def test_geometry_review_is_hash_and_scope_bound(geometry_review):
    changed = copy.deepcopy(geometry_review)
    changed["assets"]["rdkit_version"] = "unreviewed"
    with pytest.raises(ValueError, match="review"):
        structural_proxies("CCO", geometry_review=changed)


def test_partial_charge_dipole_invariance_without_rescaling():
    q = np.array([0.2, -0.1, -0.1])
    positions = np.array([[1.0, 2, 0], [0, 0, 0], [0, 1, 0]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert signed_dipole_norm(q, positions) == pytest.approx(
        signed_dipole_norm(q, positions @ rotation + 1000)
    )
    with pytest.raises(ValueError, match="neutral"):
        signed_dipole_norm(q + 0.01, positions)
