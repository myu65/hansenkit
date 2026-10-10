"""Identity and scaffold embargo used before any supervised or auxiliary fitting."""

from dataclasses import dataclass, field

import numpy as np

from .chemistry import normalize_smiles
from .splitting import (
    GROUPING_POLICY,
    core_topology_from_canonical,
    scaffold_from_canonical,
    structure_group_aliases,
)


@dataclass(frozen=True)
class HoldoutEmbargo:
    identities: frozenset[str]
    scaffolds: frozenset[str]
    series: frozenset[str] = frozenset()
    group_aliases: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self):
        aliases = {"scaffold:" + s for s in self.scaffolds}
        for s in self.identities:
            aliases.update(structure_group_aliases(normalize_smiles(s)))
        # Direct construction with only legacy scaffold reservations cannot bypass aliases.
        for scaffold in self.scaffolds - {"ACYCLIC"}:
            aliases.update(structure_group_aliases(normalize_smiles(scaffold)))
        object.__setattr__(self, "group_aliases", frozenset(aliases))

    def allows_canonical(self, canonical):
        """Internal fast path after input normalization has certified its identity."""
        if "topology:" + core_topology_from_canonical(canonical) in self.group_aliases:
            return False
        return self.group_aliases.isdisjoint(structure_group_aliases(canonical))

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
                self.allows_canonical(s) and (series is None or series[i] not in self.series)
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
            "grouping": (
                "strict Murcko plus conservative core topology and bounded tautomer aliases; "
                "all acyclic structures share one group"
            ),
            "grouping_policy": GROUPING_POLICY,
            "group_aliases": len(self.group_aliases),
        }
