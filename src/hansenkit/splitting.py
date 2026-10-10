"""Connected-component grouping prevents canonical duplicate and family leakage."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

from .chemistry import normalize_smiles
from .data import Dataset

GROUPING_POLICY = "canonical+strict-murcko+core-topology+bounded-tautomer-v1"


class TautomerGroupingError(ValueError):
    """An incomplete or unstable tautomer grouping cannot certify isolation."""


@lru_cache(maxsize=8192)
def core_topology_from_canonical(canonical: str) -> str:
    """Opaque conservative family key; not a physical SMILES or feature input.

    Remove terminal non-ring Murcko decorations and ignore bond orders/protons.
    This closes keto/enol scaffold bridges independently of which inputs are present.
    Saturated/aromatic versions of the same element-labelled core also share a group.
    """
    core = Chem.RWMol(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(canonical)))
    if not core.GetNumAtoms():
        return "ACYCLIC"
    terminals = [a.GetIdx() for a in core.GetAtoms() if a.GetDegree() == 1 and not a.IsInRing()]
    for i in reversed(terminals):
        core.RemoveAtom(i)
    for atom in core.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumRadicalElectrons(0)
    for bond in core.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(core, isomericSmiles=False)


@lru_cache(maxsize=8192)
def tautomer_parent_from_canonical(canonical: str) -> str:
    """Grouping alias only; never replace a source structure or regression input."""
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise TautomerGroupingError("Invalid structure for tautomer grouping")
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(256)
    enumerator.SetMaxTransforms(1024)
    candidates = enumerator.Enumerate(mol)
    if candidates.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
        raise TautomerGroupingError("Incomplete bounded tautomer enumeration")
    parent = Chem.MolToSmiles(enumerator.PickCanonical(candidates), isomericSmiles=True)
    try:
        return normalize_smiles(parent)
    except ValueError as exc:
        raise TautomerGroupingError("Unstable generated tautomer grouping alias") from exc


def structure_group_aliases(canonical: str, include_scaffolds=True) -> frozenset[str]:
    """Retain every legacy link and add complete tautomer identity/family aliases."""
    parent = tautomer_parent_from_canonical(canonical)
    aliases = {"identity:" + canonical, "tautomer:" + parent}
    if include_scaffolds:
        aliases.update(
            (
                "scaffold:" + scaffold_from_canonical(canonical),
                "scaffold:" + scaffold_from_canonical(parent),
                "topology:" + core_topology_from_canonical(canonical),
            )
        )
    return frozenset(aliases)


def scaffold_key(smiles: str) -> str:
    return scaffold_from_canonical(normalize_smiles(smiles))


def scaffold_from_canonical(canonical: str) -> str:
    """Internal fast path for already normalized identities; no alternate grouping policy."""
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(canonical))
    # All acyclic compounds share one conservative group; never silently use random splitting.
    return Chem.MolToSmiles(scaffold, isomericSmiles=False) or "ACYCLIC"


def grouping_keys(smiles, series=None, strategy="scaffold") -> np.ndarray:
    if strategy not in {"scaffold", "polymer_series"}:
        raise ValueError("Unknown split strategy")
    if strategy == "polymer_series" and (series is None or any(not s.strip() for s in series)):
        raise ValueError("Polymer-series splitting requires a series ID for every row")
    canonical = [normalize_smiles(s) for s in smiles]
    families = (
        [scaffold_from_canonical(s) for s in canonical] if strategy == "scaffold" else list(series)
    )
    parents = list(range(len(smiles)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, (identity, family) in enumerate(zip(canonical, families, strict=True)):
        aliases = set(structure_group_aliases(identity, include_scaffolds=strategy == "scaffold"))
        if strategy == "polymer_series":
            aliases.add("series:" + family)
        for key in sorted(aliases):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(smiles))])


def assert_group_isolation(smiles, groups, series=None, strategy="scaffold"):
    """Caller-supplied groups may merge families but must never split an alias."""
    groups = np.asarray(groups)
    if groups.shape != (len(smiles),):
        raise ValueError("One group is required per structure")
    expected = grouping_keys(smiles, series, strategy)
    assignments = {}
    for family, group in zip(expected, groups, strict=True):
        if family in assignments and assignments[family] != group:
            raise ValueError(
                "Provided groups split an identity, scaffold, series or tautomer family"
            )
        assignments[family] = group


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    groups: np.ndarray

    def report(self, dataset: Dataset) -> dict:
        output = {"grouping_policy": GROUPING_POLICY}
        for name, indices in (
            ("train", self.train),
            ("calibration", self.calibration),
            ("test", self.test),
        ):
            output[name] = {
                "rows": len(indices),
                "groups": len(set(self.groups[indices])),
                "sample_ids": [dataset.sample_ids[i] for i in indices],
            }
        return output


def split_dataset(dataset: Dataset, strategy="scaffold", seed=42) -> Split:
    groups = grouping_keys(dataset.smiles, dataset.series, strategy)
    if len(set(groups)) < 8:
        raise ValueError("Need at least eight independent groups; no random-split fallback")
    rows = np.arange(len(dataset.smiles))
    train_cal, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(rows, groups=groups)
    )
    train_rel, cal_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1).split(
            train_cal, groups=groups[train_cal]
        )
    )
    train, cal = train_cal[train_rel], train_cal[cal_rel]
    for left, right in ((train, cal), (train, test), (cal, test)):
        if set(groups[left]) & set(groups[right]):
            raise RuntimeError("Group leakage")
    return Split(train, cal, test, groups)
