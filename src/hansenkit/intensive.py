"""Size-normalized research features; no empirical HSP constants or external tables."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .chemistry import RULES, atom_coverage, normalize_smiles
from .descriptors import complete_crippen_descriptors

INTENSIVE_VERSION = "original-density-descriptors-v2-complete-crippen"
ELEMENTS = (6, 7, 8, 9, 16, 17, 35, 53, 5, 14, 15)
COUNT_NAMES = tuple(f"group.{r.name}" for r in RULES) + tuple(f"element.{z}" for z in ELEMENTS)
DENSITY_NAMES = (
    "mass_per_heavy_atom",
    "logp_per_heavy_atom",
    "refractivity_per_heavy_atom",
    "tpsa_per_heavy_atom",
    "rings_per_heavy_atom",
    "rotatable_per_heavy_atom",
    "hbd_per_heavy_atom",
    "hba_per_heavy_atom",
    "inverse_heavy_atoms",
    "inverse_sqrt_heavy_atoms",
    "unassigned_group_atom_fraction",
)
INTENSIVE_NAMES = COUNT_NAMES + tuple(f"sqrt.{n}" for n in COUNT_NAMES) + DENSITY_NAMES


@dataclass(frozen=True)
class IntensiveFeatures:
    canonical_smiles: str
    values: np.ndarray
    group_coverage_fraction: float
    unassigned_group_atoms: tuple[int, ...]


def intensive_features(smiles: str) -> IntensiveFeatures:
    """Complete explicit elemental features; incomplete functional vocabulary stays visible."""
    canonical = normalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("Intensive features require one connected molecule without ports")
    if any(a.GetAtomicNum() not in {1, *ELEMENTS} for a in mol.GetAtoms()):
        raise ValueError("Element outside the explicit research feature vocabulary")
    n = mol.GetNumHeavyAtoms()
    if n == 0:
        raise ValueError("Intensive features require heavy atoms")
    coverage = atom_coverage(canonical)
    logp, refractivity = complete_crippen_descriptors(mol)
    atoms = Counter(a.GetAtomicNum() for a in mol.GetAtoms())
    counts = (
        np.array(
            [coverage.counts[r.name] for r in RULES] + [atoms[z] for z in ELEMENTS], dtype=float
        )
        / n
    )
    descriptors = np.array(
        [
            Descriptors.MolWt(mol) / n,
            logp / n,
            refractivity / n,
            rdMolDescriptors.CalcTPSA(mol) / n,
            rdMolDescriptors.CalcNumRings(mol) / n,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / n,
            rdMolDescriptors.CalcNumHBD(mol) / n,
            rdMolDescriptors.CalcNumHBA(mol) / n,
            1 / n,
            1 / np.sqrt(n),
            len(coverage.unassigned) / n,
        ]
    )
    values = np.r_[counts, np.sqrt(counts), descriptors]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite intensive descriptors")
    return IntensiveFeatures(canonical, values, coverage.fraction, coverage.unassigned)


def is_saturated_acyclic_hydrocarbon(smiles):
    """Structural physical-zero candidate for P/H; this function only classifies structure."""
    mol = Chem.MolFromSmiles(normalize_smiles(smiles))
    return (
        len(Chem.GetMolFrags(mol)) == 1
        and mol.GetRingInfo().NumRings() == 0
        and any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        and all(
            a.GetAtomicNum() in {1, 6}
            and not a.GetFormalCharge()
            and not a.GetNumRadicalElectrons()
            for a in mol.GetAtoms()
        )
        and all(b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds())
    )
