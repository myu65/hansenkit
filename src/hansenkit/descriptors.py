"""Complete RDKit Crippen atom typing, avoiding the default substructure match cap."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, rdBase


@lru_cache(maxsize=1)
def _crippen_parameters():
    # Read the installed RDKit BSD-licensed table; no vendoring or external downloads.
    path = Path(RDConfig.RDDataDir) / "Crippen.txt"
    payload = path.read_bytes()
    rules = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            raise ValueError("Invalid installed RDKit Crippen parameter row")
        pattern = Chem.MolFromSmarts(fields[1])
        if pattern is None:
            raise ValueError("Invalid installed RDKit Crippen SMARTS")
        rules.append((pattern, float(fields[2] or 0), float(fields[3] or 0)))
    if not rules:
        raise ValueError("Empty installed RDKit Crippen parameter table")
    return tuple(rules), hashlib.sha256(payload).hexdigest()


def descriptor_provenance():
    return {"rdkit_version": rdBase.rdkitVersion, "crippen_table_sha256": _crippen_parameters()[1]}


def complete_crippen_descriptors(mol):
    """First matching atomic type in RDKit table order, including all explicit hydrogens."""
    explicit = Chem.AddHs(mol)
    done = np.zeros(explicit.GetNumAtoms(), dtype=bool)
    contributions = np.zeros((len(done), 2))
    params = Chem.SubstructMatchParameters()
    params.uniquify = False  # Query's first atom is the typed atom; retain symmetric orientations.
    params.maxMatches = 0  # RDKit's unlimited setting, independently tested beyond 1,000 matches.
    for pattern, logp, mr in _crippen_parameters()[0]:
        for match in explicit.GetSubstructMatches(pattern, params):
            atom = match[0]
            if not done[atom]:
                done[atom] = True
                contributions[atom] = logp, mr
        if done.all():
            break
    if not done.all() or not np.isfinite(contributions).all():
        raise ValueError("Incomplete or nonfinite RDKit Crippen atom typing")
    return contributions.sum(axis=0)
