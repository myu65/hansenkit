"""Connected-component grouping prevents canonical duplicate and family leakage."""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

from .chemistry import normalize_smiles
from .data import Dataset


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

    for values in (canonical, families):
        seen = {}
        for i, key in enumerate(values):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(smiles))])


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    groups: np.ndarray

    def report(self, dataset: Dataset) -> dict:
        output = {}
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
