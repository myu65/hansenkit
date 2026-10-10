import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from hansenkit.chains import assemble_homopolymer, assemble_sequence, characterize_homopolymer
from hansenkit.chemistry import atom_coverage, normalize_smiles
from hansenkit.descriptors import complete_crippen_descriptors
from hansenkit.intensive import (
    INTENSIVE_NAMES,
    intensive_features,
    is_saturated_acyclic_hydrocarbon,
)
from hansenkit.schema import EOPODistribution, PolymerInput, RepeatUnit


def polymer(smiles="*CCO*", mn=1000, cap="*O"):
    return PolymerInput(
        series_id="structural-test",
        repeat_units=[{"smiles": smiles, "mole_fraction": 1}],
        mn_g_mol=mn,
        architecture="linear",
        end_groups=[{"smiles": cap}] if cap else [],
    )


def test_connections_have_correct_formula_and_caps():
    # Three EO residues plus HO/H end caps: C6H14O4.
    smiles = assemble_homopolymer("[1*]CCO[2*]", 3, "*O")
    expected = normalize_smiles("OCCOCCOCCO")
    assert smiles == expected and "*" not in smiles and "." not in smiles
    mass = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
    assert mass == pytest.approx(3 * 44.053 + 18.015)
    assert assemble_sequence(["*CCO*", "*CC(C)O*"], "*O") == normalize_smiles("OCCOCC(C)O")


def test_connection_stereochemistry_and_alkene_geometry():
    assembled = assemble_homopolymer("[1*][C@H](F)C[2*]", 2, "*O", "*N")
    assert assembled == normalize_smiles("O[C@H](F)C[C@H](F)CN")
    double = assemble_homopolymer("[1*]C/C=C/C[2*]", 2)
    assert double == normalize_smiles("C/C=C/CC/C=C/C")


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 10**12])
def test_invalid_or_unbounded_explicit_allocation_is_prevented(count):
    with pytest.raises(ValueError):
        assemble_homopolymer("*CCO*", count)


@pytest.mark.parametrize("smiles", ["*=CC*", "**CC*", "*CC", "[1*]CC[1*]"])
def test_invalid_ports_do_not_change_bond_order_or_guess_orientation(smiles):
    with pytest.raises(ValueError):
        assemble_homopolymer(smiles, 2)


def test_schema_checks_actual_single_bonds():
    with pytest.raises(ValueError):
        RepeatUnit(smiles="*=CC*", mole_fraction=1)


def test_bulk_features_are_bounded_and_match_direct_assembly():
    result = characterize_homopolymer(polymer(mn=128 * 44.053 + 18.015))
    expected = intensive_features(assemble_homopolymer("*CCO*", 128, "*O"))
    np.testing.assert_allclose(result["features"], expected.values, atol=1e-8, rtol=1e-9)
    huge = characterize_homopolymer(polymer(mn=10**9))
    assert huge["max_explicit_atoms"] == result["max_explicit_atoms"] == 193
    assert np.isfinite(huge["features"]).all() and huge["hsp_predictions"] is None
    np.testing.assert_allclose(huge["features"], huge["bulk_limit_features"], atol=1e-3)
    assert huge["group_coverage_fraction"] == 1
    assert huge["residue_mass_g_mol"] == pytest.approx(44.053)
    assert huge["end_mass_offset_g_mol"] == pytest.approx(18.015)


def test_unknown_caps_mn_and_coverage_are_not_hidden():
    result = characterize_homopolymer(polymer(cap=None))
    assert "unknown_end_groups_hydrogen_capped_representatives" in result["issues"]
    with pytest.raises(ValueError, match="Mn"):
        characterize_homopolymer(polymer(mn=10))
    record = intensive_features("COP(=O)(O)O")
    assert record.unassigned_group_atoms and record.group_coverage_fraction < 1
    assert len(record.values) == len(INTENSIVE_NAMES) and np.isfinite(record.values).all()


def test_hydrocarbon_classifier_does_not_override_other_chemistry():
    assert is_saturated_acyclic_hydrocarbon("CC(C)CCCC")
    for s in ("C1CCCCC1", "c1ccccc1", "C=CC", "CCO", "CC.[Na+]", "[CH3]"):
        assert not is_saturated_acyclic_hydrocarbon(s)


def test_sequence_iterators_are_bounded_before_materialization():
    def unbounded():
        for index in range(1000):
            assert index <= 256, "Assembly consumed an unbounded sequence"
            yield "*CC*"

    with pytest.raises(ValueError, match="Explicit assembly"):
        assemble_sequence(unbounded())


def test_eo_metadata_does_not_invent_identity_for_other_repeats():
    arbitrary = polymer("*CC*", cap=None).model_copy(
        update={"eo_po": EOPODistribution(eo_mean=20, po_mean=0)}
    )
    report = characterize_homopolymer(arbitrary)
    assert "eo_po_metadata_not_mapped_to_repeat_unit" in report["issues"]
    assert "eo_po_mean_and_mn_inconsistent" not in report["issues"]


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)OC", "c1ccccc1", "C[N+](C)(C)C"])
def test_complete_crippen_preserves_small_molecule_descriptor_values(smiles):
    mol = Chem.MolFromSmiles(smiles)
    np.testing.assert_allclose(
        complete_crippen_descriptors(mol),
        [Descriptors.MolLogP(mol), Descriptors.MolMR(mol)],
        atol=1e-10,
    )


def test_crippen_and_group_counts_do_not_silently_stop_at_1000_matches():
    single = Chem.MolFromSmiles("C")
    replicas = Chem.MolFromSmiles(".".join(["C"] * 1001))
    expected = np.array([Descriptors.MolLogP(single), Descriptors.MolMR(single)]) * 1001
    np.testing.assert_allclose(complete_crippen_descriptors(replicas), expected, atol=1e-7)
    coverage = atom_coverage("C" * 1001)
    assert coverage.counts["saturated_c"] == 1001
    assert coverage.fraction == 1


@pytest.mark.parametrize(
    "unit", ["[1*]CC(c1ccccc1)[2*]", "[1*]CC(C)(C(=O)OC)[2*]", "[1*]NCCCCCC(=O)[2*]"]
)
def test_representative_bulk_additivity_matches_independent_128_unit_chain(unit):
    direct = assemble_homopolymer(unit, 128)
    mn = Descriptors.MolWt(Chem.MolFromSmiles(direct))
    record = characterize_homopolymer(polymer(unit, mn=mn, cap=None))
    np.testing.assert_allclose(record["features"], intensive_features(direct).values, atol=1e-7)
