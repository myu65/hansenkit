import pytest

from hansenkit.chemistry import atom_coverage, chemical_features, normalize_smiles
from hansenkit.schema import MoleculeInput
from hansenkit.scope import assess_scope


def test_normalization_preserves_chemical_identity():
    assert normalize_smiles("OCC") == normalize_smiles("[CH3:1][CH2:2][OH:3]") == "CCO"
    assert normalize_smiles("[H]OC") == "CO"
    assert normalize_smiles("C[C@H](O)F") != normalize_smiles("C[C@@H](O)F")
    assert "." in normalize_smiles("CCO.[Na+]")
    assert "+" in normalize_smiles("C[N+](C)(C)C")
    assert "13" in normalize_smiles("[13CH3]CO")


@pytest.mark.parametrize("smiles", ["", "  ", "not_smiles", "C1CCC", "C(C)(C)(C)(C)C"])
def test_invalid_smiles(smiles):
    with pytest.raises(ValueError):
        normalize_smiles(smiles)


@pytest.mark.parametrize(
    "smiles,group,saturated",
    [
        ("CC(=O)OC", "ester", 2),
        ("CC(=O)N", "amide", 1),
        ("CC(=O)O", "carboxylic_acid", 1),
        ("CC(=O)C", "carbonyl", 2),
    ],
)
def test_owned_group_atoms_exclude_context(smiles, group, saturated):
    coverage = atom_coverage(smiles)
    assert coverage.fraction == 1
    assert not coverage.unassigned
    assert coverage.counts[group] == 1
    assert coverage.counts["saturated_c"] == saturated
    assert len(coverage.owners) == coverage.heavy_atoms
    if group != "carbonyl":
        assert coverage.counts["carbonyl"] == 0


def test_unassigned_atoms_are_not_hidden_by_generic_fallback():
    coverage = atom_coverage("COP(=O)(O)O")
    assert coverage.unassigned
    assert coverage.fraction < 1
    with pytest.raises(ValueError, match="Unassigned"):
        chemical_features("COP(=O)(O)O")


@pytest.mark.parametrize(
    "smiles,reason",
    [
        ("CCO.[Na+]", "mixture_or_salt_not_validated"),
        ("C[N+](C)(C)C", "ionic_not_validated"),
        ("[NH3+]CC(=O)[O-]", "ionic_not_validated"),
        ("*CCO*", "repeat_unit_not_validated"),
        ("CCCCCCCCOCCOCCOCCO", "eo_po_chain_not_validated"),
        ("C" * 65, "size_outside_poc_scope"),
        ("[CH3]", "radical_not_validated"),
    ],
)
def test_unsupported_scope(smiles, reason):
    result = assess_scope(MoleculeInput(smiles=smiles))
    assert not result.supported
    assert reason in result.reasons


def test_supported_molecule_and_temperature():
    assert assess_scope(MoleculeInput(smiles="CCO")).supported
    assert not assess_scope(MoleculeInput(smiles="CCO", temperature_k=310)).supported
