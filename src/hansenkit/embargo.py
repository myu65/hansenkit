"""Identity and scaffold embargo used before any supervised or auxiliary fitting."""

from dataclasses import dataclass

import numpy as np

from .chemistry import normalize_smiles
from .splitting import scaffold_from_canonical


@dataclass(frozen=True)
class HoldoutEmbargo:
    identities: frozenset[str]
    scaffolds: frozenset[str]
    series: frozenset[str] = frozenset()

    @classmethod
    def from_smiles(cls, smiles, series=()):
        canonical = frozenset(normalize_smiles(s) for s in smiles)
        return cls(
            canonical, frozenset(scaffold_from_canonical(s) for s in canonical), frozenset(series)
        )

    def eligible(self, smiles, series=None):
        if series is not None and len(series) != len(smiles):
            raise ValueError("One series identifier is required per structure")
        canonical = [normalize_smiles(s) for s in smiles]
        return np.array(
            [
                s not in self.identities
                and scaffold_from_canonical(s) not in self.scaffolds
                and (series is None or series[i] not in self.series)
                for i, s in enumerate(canonical)
            ],
            dtype=bool,
        )

    def assert_disjoint(self, smiles, series=None):
        if not self.eligible(smiles, series).all():
            raise ValueError("Reserved holdout identity, scaffold or series entered fitting data")

    def report(self):
        return {
            "identities": len(self.identities),
            "scaffolds": len(self.scaffolds),
            "series": len(self.series),
            "grouping": "strict Murcko; all acyclic structures share one group",
        }
