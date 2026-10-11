"""Authored graph fixtures; these tests do not qualify material or HSP accuracy."""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler, rdMolDescriptors

from hansenkit.chains import assemble_homopolymer, assemble_sequence
from hansenkit.chemistry import normalize_smiles


def specified_double_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return sum(
        bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
        for bond in mol.GetBonds()
    )


@pytest.mark.parametrize("n", [1, 2, 4, 8])
@pytest.mark.parametrize("caps", [("*C", "*C"), ("*[H]", "*O")])
@pytest.mark.parametrize(
    "unit",
    [
        "*/C=C/C=C/*",
        "*/C=C/C=C\\*",
        "*\\C=C/C=C/*",
        "*\\C=C/C=C\\*",
        "[1*]/C=C/C=C/[2*]",
        "[1*]/C=C/C=C\\[2*]",
    ],
)
def test_join_preserves_all_specified_polyene_bonds(unit, n, caps, capfd):
    output = assemble_homopolymer(unit, n, *caps)
    # An H cap makes the first alkene carbon CH2, removing its stereo center.
    assert specified_double_bonds(output) == 2 * n - int(caps[0] == "*[H]")
    assert normalize_smiles(output) == output
    assert "inconsistent bond directions" not in capfd.readouterr().err


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("[1*]/C=C/C=C/[2*]", "C/C=C/C=C/C=C/C=C/C"),
        ("[1*]/C=C/C=C\\[2*]", "C/C=C/C=C\\C=C\\C=C/C"),
    ],
)
def test_two_polyene_units_match_independently_authored_geometry(unit, expected):
    actual = assemble_homopolymer(unit, 2, "*C", "*C")
    assert actual == normalize_smiles(expected)
    assert rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(actual)) == "C10H14"


def test_mixed_polyene_directions_and_port_orientation_are_preserved():
    actual = assemble_sequence(["[1*]/C=C/C=C/[2*]", "[1*]/C=C/C=C\\[2*]"], "*C", "*C")
    expected = normalize_smiles("C/C=C/C=C/C=C/C=C\\C")
    assert actual == expected


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_methyl_termination_removes_only_the_degenerate_stereo_center(n):
    actual = assemble_homopolymer("[1*]/C(C)=C/[2*]", n, "*C", "*C")
    assert specified_double_bonds(actual) == n - 1


def test_changed_cip_priority_retains_the_original_atom_reference_relation():
    source = Chem.MolFromSmiles("*/C=C(F)/*")
    rdCIPLabeler.AssignCIPLabels(source)
    source_bond = next(b for b in source.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    actual = assemble_homopolymer("*/C=C(F)/*", 1, "*C", "*C")
    assert actual == normalize_smiles("C/C=C(F)/C")
    capped = Chem.MolFromSmiles(actual)
    rdCIPLabeler.AssignCIPLabels(capped)
    capped_bond = next(b for b in capped.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    assert source_bond.GetProp("_CIPCode") == "E"
    assert capped_bond.GetProp("_CIPCode") == "Z"


@pytest.mark.parametrize("n", [1, 2, 4, 8])
@pytest.mark.parametrize("tag", ["@", "@@"])
def test_existing_tetrahedral_neighbor_parity_survives_join(n, tag):
    actual = assemble_homopolymer(f"[1*][C{tag}H](F)C[2*]", n, "*O", "*N")
    expected = normalize_smiles("O" + f"[C{tag}H](F)C" * n + "N")
    assert actual == expected


@pytest.mark.parametrize("n", [1, 2, 4, 8])
@pytest.mark.parametrize("tag", ["@", "@@"])
@pytest.mark.parametrize("right", ["*O", "*[H]"])
def test_hydrogen_cap_removes_only_the_degenerate_terminal_tetrahedral_center(n, tag, right):
    actual = assemble_homopolymer(f"[1*][C{tag}H](F)C[2*]", n, "*[H]", right)
    expected = normalize_smiles(
        "FC" + "C" + f"[C{tag}H](F)C" * (n - 1) + ("O" if right == "*O" else "")
    )
    assert actual == expected
    tagged = sum(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in Chem.MolFromSmiles(actual).GetAtoms()
    )
    assert tagged == n - 1


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_separated_unspecified_double_bonds_remain_unspecified(n):
    actual = assemble_homopolymer("[1*]CC=CC/C=C/[2*]", n, "*C", "*C")
    assert specified_double_bonds(actual) == n
    total = sum(
        b.GetBondType() == Chem.BondType.DOUBLE for b in Chem.MolFromSmiles(actual).GetBonds()
    )
    assert total == 2 * n


@pytest.mark.parametrize("unit", ["*C=C/C=C/*", "*C=C/C=C\\*"])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("caps", [("*C", "*C"), ("*[H]", "*O")])
def test_unrepresentable_partial_polyene_refuses_an_invented_assignment(unit, n, caps):
    with pytest.raises(ValueError, match="unspecified double bond acquired stereo"):
        assemble_homopolymer(unit, n, *caps)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_nonclassic_double_bond_flags_retain_opaque_source_encoding_only(n):
    # RDKit accepts this P=N syntax but its potential-stereo classifier omits it.
    # Retaining source flags does not validate physical E/Z or any material property.
    actual = assemble_homopolymer("*/P=N/*", n, "*C", "*C")
    assert specified_double_bonds(actual) == n
    assert normalize_smiles(actual) == actual
    if n == 1:
        assert actual == normalize_smiles("C/P=N/C")
