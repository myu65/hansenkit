from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors

from .chemistry import atom_coverage, molecule
from .schema import ChemicalInput


@dataclass(frozen=True)
class Applicability:
    supported: bool
    reasons: tuple[str, ...]


def assess_scope(record: ChemicalInput) -> Applicability:
    if record.kind != "molecule":
        return Applicability(False, (f"{record.kind}_not_validated",))
    mol = molecule(record.smiles)
    reasons = []
    if len(Chem.GetMolFrags(mol)) > 1:
        reasons.append("mixture_or_salt_not_validated")
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        reasons.append("ionic_not_validated")
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        reasons.append("repeat_unit_not_validated")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        reasons.append("radical_not_validated")
    if any(a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53} for a in mol.GetAtoms()):
        reasons.append("element_not_supported")
    if mol.GetNumHeavyAtoms() > 60 or Descriptors.MolWt(mol) > 500:
        reasons.append("size_outside_poc_scope")
    # Conservative screening of repeated ether chains, including PO methyl substitution.
    repeat = Chem.MolFromSmarts("[OX2][CX4][CX4][OX2][CX4][CX4][OX2]")
    if mol.HasSubstructMatch(repeat):
        reasons.append("eo_po_chain_not_validated")
    coverage = atom_coverage(record.smiles)
    if coverage.fraction < 1 or coverage.unassigned:
        reasons.append("incomplete_atom_coverage")
    if abs(record.temperature_k - 298.15) > 1e-6:
        reasons.append("temperature_not_validated")
    return Applicability(not reasons, tuple(reasons))
