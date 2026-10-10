import pytest

from hansenkit.periodic import periodic_topology_alias


@pytest.mark.parametrize(
    "representations",
    [
        ["*CO*", "*OC*", "*COCO*", "*" + "CO" * 7 + "*"],
        ["*C*", "*CC*", "*CCCC*"],
        ["*CCO*", "*COC*", "*OCC*", "*CCOCCO*"],
        ["[1*]CC(C)[2*]", "[2*]C(C)C[1*]", "*CC(C)CC(C)*"],
    ],
)
def test_periodic_alias_invariant_to_cut_ports_and_superunit(representations):
    assert len({periodic_topology_alias(s) for s in representations}) == 1


def test_periodic_alias_preserves_element_and_branch_topology():
    assert periodic_topology_alias("*CO*") != periodic_topology_alias("*CN*")
    assert periodic_topology_alias("*CCC*") != periodic_topology_alias("*CC(C)*")
    # A deliberately conservative merge does not alter the actual source SMILES.
    assert periodic_topology_alias("*CC*\n") == periodic_topology_alias("*C=C*")
    assert periodic_topology_alias("*C([2H])C*") == periodic_topology_alias("*CC*")


@pytest.mark.parametrize(
    "smiles", ["", "CC", "*C", "*C.*C", "*C(*)*", "*=CC*", "**", "*" + "C" * 513 + "*"]
)
def test_periodic_alias_refuses_undefined_or_unbounded_graph(smiles):
    with pytest.raises(ValueError):
        periodic_topology_alias(smiles)
