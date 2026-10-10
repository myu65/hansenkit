"""Conservative periodic topology alias, never a molecule or prediction feature."""

import hashlib
import json
import math
from collections import Counter
from functools import reduce

from rdkit import Chem

PERIODIC_POLICY = "two-port-topology-wl8-normalized-v1"


def periodic_topology_alias(repeat_unit: str) -> str:
    """Normalized local colors on a periodic quotient; graph collisions merge groups.

    The virtual closing edge may be parallel or a loop. Count both incidences so
    equivalent primitive/superunit/cut representations yield identical colors.
    Ignore bond order, proton placement, isotope, charge and stereo for grouping
    only. These remain untouched in actual feature/encoder structures.
    """
    if not isinstance(repeat_unit, str) or not repeat_unit.strip() or len(repeat_unit) > 8192:
        raise ValueError("Periodic grouping needs a bounded nonempty repeat SMILES")
    with Chem.rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(repeat_unit)
    if mol is None or mol.GetNumAtoms() > 512 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Invalid, oversized or disconnected periodic repeat")
    hydrogen_policy = Chem.RemoveHsParameters()
    hydrogen_policy.removeIsotopes = True
    mol = Chem.RemoveHs(mol, hydrogen_policy)
    ports = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(ports) != 2 or any(
        a.GetDegree() != 1
        or a.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE
        or a.GetNeighbors()[0].GetAtomicNum() == 0
        for a in ports
    ):
        raise ValueError("Periodic grouping needs two singly bonded attachment ports")
    real = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    if not real:
        raise ValueError("Periodic grouping needs real atoms")
    indices = {a.GetIdx(): i for i, a in enumerate(real)}
    neighbors = [[] for _ in real]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in indices and b in indices:
            neighbors[indices[a]].append(indices[b])
            neighbors[indices[b]].append(indices[a])
    a, b = [indices[p.GetNeighbors()[0].GetIdx()] for p in ports]
    neighbors[a].append(b)
    neighbors[b].append(a)
    colors = [f"{atom.GetAtomicNum()}:{len(neighbors[i])}" for i, atom in enumerate(real)]
    for _ in range(8):
        colors = [
            hashlib.sha256(
                json.dumps([colors[i], sorted(colors[j] for j in adjacent)]).encode()
            ).hexdigest()
            for i, adjacent in enumerate(neighbors)
        ]
    populations = Counter(colors)
    divisor = reduce(math.gcd, populations.values())
    normalized = sorted((color, count // divisor) for color, count in populations.items())
    digest = hashlib.sha256(json.dumps(normalized).encode()).hexdigest()
    return f"periodic:{PERIODIC_POLICY}:{digest}"
