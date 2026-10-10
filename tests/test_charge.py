import pytest
from rdkit import Chem

from hansenkit.charge import charge_classification
from hansenkit.schema import MoleculeInput
from hansenkit.scope import assess_scope


@pytest.mark.parametrize(
    "smiles",
    [
        "C[N+](=O)[O-]",
        "O=[N+]([O-])c1ccccc1",
        "O=[N+]([O-])c1ccc([N+](=O)[O-])cc1",
        "CO[N+](=O)[O-]",
        "O=[N+]([O-])OCCO[N+](=O)[O-]",
    ],
)
def test_covalent_nitro_and_nitrate_preserve_charge_and_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    before = Chem.MolToSmiles(mol)
    assert charge_classification(mol) == "neutral_covalent_nitro_or_nitrate"
    assert Chem.MolToSmiles(mol) == before
    assert any(a.GetFormalCharge() for a in mol.GetAtoms())


@pytest.mark.parametrize(
    "smiles",
    [
        "[NH3+]CC(=O)[O-]",
        "C[N+](C)(C)C.[Cl-]",
        "[Na+].[O-]C=O",
        "C[NH3+]",
        "C[N+]1([O-])CCOCC1",
        "CCN=[N+]=[N-]",
        "[C-]#[N+]CC",
        "[NH3+]CC([N+](=O)[O-])C(=O)[O-]",
        "O=[N+]([O-])O",
    ],
)
def test_net_zero_or_one_recognized_group_does_not_qualify_all_charges(smiles):
    assert charge_classification(Chem.MolFromSmiles(smiles)) == "unreviewed_charge_or_fragments"


def test_charge_classification_does_not_expand_default_hsp_scope():
    assert charge_classification(Chem.MolFromSmiles("CCO")) == "no_formal_charge"
    assert not assess_scope(MoleculeInput(smiles="C[N+](=O)[O-]")).supported
