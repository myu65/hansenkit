"""Bounded structural proxies, never liquid H-bond energies or HSP components.

No external downloads, graph neutralization, force-field fallback or implicit 3D run.
The optional conformer calculation requires a hash-bound local-use review.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures, Descriptors, rdForceFieldHelpers

from .charge import charge_classification
from .chemistry import normalize_smiles

TOPOLOGY_NAMES = (
    "donor_sites_per_heavy",
    "acceptor_sites_per_heavy",
    "donor_acceptor_product_per_heavy_squared",
    "donor_acceptor_graph_distance_3_to_7_per_heavy_squared",
)
CHARGE_NAMES = (
    "gasteiger_abs_charge_per_heavy",
    "gasteiger_charge_squared_per_heavy",
    "gasteiger_max_charge",
    "gasteiger_min_charge",
    "gasteiger_donor_h_charge_sum_per_heavy",
    "gasteiger_acceptor_charge_sum_per_heavy",
    "gasteiger_hetero_abs_charge_per_heavy",
)
GEOMETRY_NAMES = (
    "conformer_volume_angstrom3_per_heavy_mean",
    "conformer_volume_angstrom3_per_heavy_std",
    "mmff_dipole_e_angstrom_mean",
    "mmff_dipole_e_angstrom_std",
    "mmff_dipole_squared_per_volume_e2_per_angstrom_mean",
    "radius_of_gyration_angstrom_mean",
    "close_donor_acceptor_pairs_per_heavy_mean",
    "mmff_intramolecular_energy_range_kcal_per_mol",
)
FEATURE_NAMES = TOPOLOGY_NAMES + CHARGE_NAMES + GEOMETRY_NAMES
CONFORMER_PROTOCOL = {
    "method": "ETKDGv3_then_MMFF94",
    "conformers": 3,
    "embedding_max_iterations": 1000,
    "embedding_timeout_seconds": 10,
    "optimization_max_iterations": 600,
    "threads": 1,
    "pruning": False,
    "maximum_heavy_atoms": 60,
    "maximum_molecular_weight": 500.0,
    "all_conformers_must_converge": True,
    "fallback": None,
    "equilibrium_or_physical_accuracy_qualified": False,
}


def runtime_assets():
    """Hash installed parameter-bearing binaries and chemical feature definitions."""
    root = Path(Chem.__file__).resolve().parents[1]
    files = [root / "license.txt", Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"]
    for name in ("rdPartialCharges", "rdForceFieldHelpers", "rdDistGeom", "rdMolChemicalFeatures"):
        files.extend(root.glob(f"Chem/{name}.*"))
    # Shared libraries may hold parameter tables instead of the Python extension.
    files.extend((root.parent / "rdkit.libs").glob("*"))
    files = sorted({p.resolve() for p in files if p.is_file()})
    return {
        "rdkit_version": rdBase.rdkitVersion,
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


@lru_cache(maxsize=1)
def _factory():
    return ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))


def _sites(mol):
    features = _factory().GetFeaturesForMol(mol)
    return tuple(
        sorted({i for f in features if f.GetFamily() == family for i in f.GetAtomIds()})
        for family in ("Donor", "Acceptor")
    )


def signed_dipole_norm(charges, positions):
    """Neutral partial-charge conformer dipole in e Å, not a condensed-phase response."""
    charges, positions = np.asarray(charges, float), np.asarray(positions, float)
    if (
        charges.ndim != 1
        or positions.shape != (len(charges), 3)
        or not len(charges)
        or not np.isfinite(charges).all()
        or not np.isfinite(positions).all()
        or abs(charges.sum()) > 1e-5
    ):
        raise ValueError("Require finite complete positions and neutral partial charges")
    return float(np.linalg.norm(charges @ (positions - positions.mean(0))))


@dataclass(frozen=True)
class StructuralProxies:
    canonical_smiles: str
    values: np.ndarray
    diagnostics: dict

    @property
    def names(self):
        return FEATURE_NAMES


def _geometry(mol, donors, acceptors, review):
    if (
        not isinstance(review, dict)
        or review.get("assets") != runtime_assets()
        or review.get("protocol") != CONFORMER_PROTOCOL
        or review.get("audit_basis") != "operator_assumption"
        or review.get("scope") != "local_scientific_private_noncommercial"
        or review.get("geometry_execution_allowed") is not True
        or review.get("redistribution_allowed") is not False
    ):
        raise ValueError("Exact local geometry assets, protocol and scoped review required")
    diagnostic = {"status": "size_outside_protocol", "protocol": CONFORMER_PROTOCOL.copy()}
    missing = np.full(len(GEOMETRY_NAMES), np.nan)
    if (
        mol.GetNumHeavyAtoms() > CONFORMER_PROTOCOL["maximum_heavy_atoms"]
        or Descriptors.MolWt(mol) > CONFORMER_PROTOCOL["maximum_molecular_weight"]
    ):
        return missing, diagnostic
    explicit = Chem.AddHs(Chem.Mol(mol))
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(explicit):
        diagnostic["status"] = "missing_mmff_parameters"
        return missing, diagnostic
    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(explicit, mmffVariant="MMFF94")
    if properties is None:
        diagnostic["status"] = "missing_mmff_properties"
        return missing, diagnostic
    charges = np.array([properties.GetMMFFPartialCharge(i) for i in range(explicit.GetNumAtoms())])
    if not np.isfinite(charges).all() or abs(charges.sum()) > 1e-5:
        diagnostic["status"] = "invalid_mmff_partial_charges"
        return missing, diagnostic
    original = Chem.MolToSmiles(mol, isomericSmiles=True)
    seed = int(hashlib.sha256(original.encode()).hexdigest()[:8], 16) % (2**31 - 1)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.pruneRmsThresh = -1.0
    parameters.maxIterations = CONFORMER_PROTOCOL["embedding_max_iterations"]
    parameters.timeout = CONFORMER_PROTOCOL["embedding_timeout_seconds"]
    diagnostic["seed"] = seed
    try:
        ids = list(AllChem.EmbedMultipleConfs(explicit, numConfs=3, params=parameters))
        diagnostic["embedded_conformer_ids"] = ids
        if len(ids) != 3:
            diagnostic["status"] = "incomplete_embedding"
            return missing, diagnostic
        optimized = list(
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                explicit, numThreads=1, maxIters=600, mmffVariant="MMFF94"
            )
        )
        diagnostic["optimization"] = [{"status": int(s), "energy": float(e)} for s, e in optimized]
        diagnostic["coordinates_angstrom"] = [
            explicit.GetConformer(i).GetPositions().tolist() for i in ids
        ]
        diagnostic["mmff_partial_charges"] = charges.tolist()
        if len(optimized) != 3 or any(s != 0 or not np.isfinite(e) for s, e in optimized):
            diagnostic["status"] = "incomplete_optimization"
            return missing, diagnostic
        rows = []
        for i in ids:
            positions = explicit.GetConformer(i).GetPositions()
            volume = float(AllChem.ComputeMolVolume(explicit, confId=i))
            if not np.isfinite(positions).all() or not np.isfinite(volume) or volume <= 0:
                diagnostic["status"] = "invalid_geometry"
                return missing, diagnostic
            dipole = signed_dipole_norm(charges, positions)
            close = sum(
                d != a and np.linalg.norm(positions[d] - positions[a]) <= 3.5
                for d in donors
                for a in acceptors
            )
            heavy_positions = positions[: mol.GetNumAtoms()]
            radius = np.sqrt(np.mean(np.sum((heavy_positions - heavy_positions.mean(0)) ** 2, 1)))
            rows.append(
                (volume / mol.GetNumHeavyAtoms(), dipole, dipole**2 / volume, radius, close)
            )
        rows = np.array(rows)
        energies = np.array([e for _, e in optimized])
        values = np.array(
            [
                rows[:, 0].mean(),
                rows[:, 0].std(),
                rows[:, 1].mean(),
                rows[:, 1].std(),
                rows[:, 2].mean(),
                rows[:, 3].mean(),
                rows[:, 4].mean() / mol.GetNumHeavyAtoms(),
                np.ptp(energies),
            ]
        )
        diagnostic["status"] = "complete_fixed_protocol"
        return values, diagnostic
    except (ValueError, RuntimeError):
        diagnostic["status"] = "geometry_execution_failure"
        return missing, diagnostic


def structural_proxies(smiles, *, geometry_review=None):
    """Complete site/charge blocks and optional all-or-missing conformer block.

    A failed block stays NaN with a reason. There is no molecule exclusion based
    on labels and no numeric HSP fallback. Charge models are kept distinct.
    """
    canonical = normalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if (
        charge_classification(mol) == "unreviewed_charge_or_fragments"
        or not 1 <= mol.GetNumHeavyAtoms() <= 256
        or any(
            a.GetNumRadicalElectrons() or a.GetIsotope() or not a.GetAtomicNum()
            for a in mol.GetAtoms()
        )
    ):
        raise ValueError("Structural proxies require a reviewed neutral connected graph")
    donors, acceptors = _sites(mol)
    n = mol.GetNumHeavyAtoms()
    distance = Chem.GetDistanceMatrix(mol)
    topology = np.array(
        [
            len(donors) / n,
            len(acceptors) / n,
            len(donors) * len(acceptors) / n**2,
            sum(3 <= distance[d, a] <= 7 for d in donors for a in acceptors) / n**2,
        ]
    )
    explicit = Chem.AddHs(Chem.Mol(mol))
    charge = np.full(len(CHARGE_NAMES), np.nan)
    diagnostic = {
        "qualified_hsp_backend": False,
        "hydrogen_bond_strength_measured": False,
        "charge_model": "Gasteiger_nIter12_throwOnParamFailure",
        "donor_atoms": list(donors),
        "acceptor_atoms": list(acceptors),
        "rdkit_version": rdBase.rdkitVersion,
    }
    try:
        AllChem.ComputeGasteigerCharges(explicit, nIter=12, throwOnParamFailure=True)
        q = np.array([float(a.GetProp("_GasteigerCharge")) for a in explicit.GetAtoms()])
        if not np.isfinite(q).all() or abs(q.sum()) > 1e-5:
            raise ValueError("Incomplete or nonneutral Gasteiger charges")
        donor_h = {
            h.GetIdx()
            for d in donors
            for h in explicit.GetAtomWithIdx(d).GetNeighbors()
            if h.GetAtomicNum() == 1
        }
        hetero = [a.GetIdx() for a in explicit.GetAtoms() if a.GetAtomicNum() not in (1, 6)]
        charge = np.array(
            [
                np.abs(q).sum() / n,
                (q**2).sum() / n,
                q.max(),
                q.min(),
                q[list(donor_h)].sum() / n,
                q[list(acceptors)].sum() / n,
                np.abs(q[hetero]).sum() / n,
            ]
        )
        diagnostic.update(charge_status="complete", gasteiger_partial_charges=q.tolist())
    except (ValueError, RuntimeError):
        diagnostic["charge_status"] = "missing_or_invalid_parameters"
    geometry = np.full(len(GEOMETRY_NAMES), np.nan)
    geometry_diagnostic = {"status": "not_requested"}
    if geometry_review is not None:
        geometry, geometry_diagnostic = _geometry(mol, donors, acceptors, geometry_review)
    diagnostic["geometry"] = geometry_diagnostic
    return StructuralProxies(canonical, np.r_[topology, charge, geometry], diagnostic)
