"""Structural charge classification, without neutralizing or qualifying HSP output."""

from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=1)
def _neutral_charge_motif():
    # Nitro C-N and nitrate ester C-O-N: retain their required charge-separated SMILES.
    return Chem.MolFromSmarts("[N+X3](=[OX1+0])([O-X1])[$([#6+0]),$([OX2+0][#6+0])]")


def charge_classification(mol: Chem.Mol) -> str:
    """Recognize two neutral covalent motifs; other charges remain unreviewed.

    Net zero alone does not certify neutrality: zwitterions, salts, N-oxides,
    azides and isocyanides are deliberately not inferred from this classification.
    The caller must still check radicals, isotope, coverage and physical applicability.
    """
    if len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        return "unreviewed_charge_or_fragments"
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    if not charged:
        return "no_formal_charge"
    recognized = set()
    for match in mol.GetSubstructMatches(_neutral_charge_motif(), maxMatches=0):
        recognized.update((match[0], match[2]))
    if charged == recognized:
        return "neutral_covalent_nitro_or_nitrate"
    return "unreviewed_charge_or_fragments"
