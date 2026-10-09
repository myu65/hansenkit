"""Original SMARTS features: no HSP group coefficients or third-party numerical tables."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def normalize_smiles(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    with Chem.rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES")
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Preserve stereo, charge, isotopes and every fragment; do not silently desalt/neutralize.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def molecule(smiles: str) -> Chem.Mol:
    return Chem.MolFromSmiles(normalize_smiles(smiles))


@dataclass(frozen=True)
class GroupRule:
    name: str
    smarts: str
    owned_query_atoms: tuple[int, ...]


# Priority and ownership distinguish a chemical group from its surrounding context.
# For an ester, the alkoxy carbon is context and remains available to the carbon rule.
RULES = (
    GroupRule("amide", "[CX3](=[OX1])[NX3]", (0, 1, 2)),
    GroupRule("carboxylic_acid", "[CX3](=[OX1])[OX2H1]", (0, 1, 2)),
    GroupRule("ester", "[CX3](=[OX1])[OX2H0][#6]", (0, 1, 2)),
    GroupRule("sulfone", "[SX4](=[OX1])(=[OX1])", (0, 1, 2)),
    GroupRule("sulfoxide", "[SX3](=[OX1])", (0, 1)),
    GroupRule("nitrile", "[CX2]#[NX1]", (0, 1)),
    GroupRule("carbonyl", "[CX3]=[OX1]", (0, 1)),
    GroupRule("phenol", "[OX2H1][c]", (0,)),
    GroupRule("alcohol", "[OX2H1][CX4]", (0,)),
    GroupRule("ether", "[OX2H0]([#6])[#6]", (0,)),
    GroupRule("amine", "[NX3;!$(N[C,S,P]=[O,S,N]);!$(N=N)]", (0,)),
    GroupRule("aromatic_n", "[n]", (0,)),
    GroupRule("aromatic_o", "[o]", (0,)),
    GroupRule("aromatic_s", "[s]", (0,)),
    GroupRule("thiol", "[SX2H1]", (0,)),
    GroupRule("sulfide", "[SX2H0]([#6])[#6]", (0,)),
    GroupRule("halogen", "[F,Cl,Br,I]", (0,)),
    GroupRule("aromatic_c", "[c]", (0,)),
    GroupRule("saturated_c", "[CX4]", (0,)),
    GroupRule("unsaturated_c", "[C;X2,X3]", (0,)),
)
DESCRIPTOR_NAMES = (
    "molecular_weight",
    "heavy_atoms",
    "rings",
    "rotatable_bonds",
    "hbond_donors",
    "hbond_acceptors",
    "tpsa",
)
FEATURE_NAMES = tuple(rule.name for rule in RULES) + DESCRIPTOR_NAMES
FEATURE_VERSION = "original-owned-smarts-v1"


@lru_cache(maxsize=1)
def queries() -> tuple[Chem.Mol, ...]:
    return tuple(Chem.MolFromSmarts(rule.smarts) for rule in RULES)


@dataclass(frozen=True)
class Coverage:
    canonical_smiles: str
    counts: dict[str, int]
    owners: dict[int, str]
    unassigned: tuple[int, ...]
    heavy_atoms: int

    @property
    def fraction(self) -> float:
        return len(self.owners) / self.heavy_atoms if self.heavy_atoms else 0.0


def atom_coverage(smiles: str) -> Coverage:
    canonical = normalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    owners: dict[int, str] = {}
    counts = dict.fromkeys((r.name for r in RULES), 0)
    for rule, query in zip(RULES, queries(), strict=True):
        for match in sorted(mol.GetSubstructMatches(query, uniquify=True)):
            owned = tuple(match[i] for i in rule.owned_query_atoms)
            if any(idx in owners for idx in owned):
                continue
            counts[rule.name] += 1
            owners.update(dict.fromkeys(owned, rule.name))
    heavy = tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    missing = tuple(i for i in heavy if i not in owners)
    return Coverage(canonical, counts, owners, missing, len(heavy))


def chemical_features(smiles: str) -> np.ndarray:
    coverage = atom_coverage(smiles)
    if coverage.unassigned or coverage.fraction < 1:
        raise ValueError("Unassigned heavy atoms: chemical feature vocabulary is incomplete")
    mol = Chem.MolFromSmiles(coverage.canonical_smiles)
    descriptors = (
        Descriptors.MolWt(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
    )
    return np.array([coverage.counts[r.name] for r in RULES] + list(descriptors), dtype=float)
