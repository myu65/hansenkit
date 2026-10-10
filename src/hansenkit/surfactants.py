"""Explicit nonionic EO/PO block assembly and population moments; no automatic HSP values."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from .chains import _fragment, _join, assemble_sequence
from .chemistry import normalize_smiles
from .descriptors import descriptor_provenance
from .intensive import COUNT_NAMES, INTENSIVE_VERSION, intensive_features
from .schema import SurfactantInput

EO = "[1*]OCC[2*]"
PO = "[1*]OCC(C)[2*]"


def assemble_blocks(tail, head, eo, po, order="eo_then_po"):
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (eo, po)):
        raise ValueError("Explicit EO/PO counts must be nonnegative integers")
    if order not in {"eo_then_po", "po_then_eo"} or eo + po > 256:
        raise ValueError("Explicit block order or repeat count is invalid")
    if eo + po == 0:
        mol = _join(_fragment(tail, 1), _fragment(head, 1), "cap", "cap")
        return normalize_smiles(Chem.MolToSmiles(mol, isomericSmiles=True))
    units = (EO,) * eo + (PO,) * po if order == "eo_then_po" else (PO,) * po + (EO,) * eo
    return assemble_sequence(units, tail, head)


def _raw(smiles):
    record = intensive_features(smiles)
    mol = Chem.MolFromSmiles(smiles)
    h = mol.GetNumHeavyAtoms()
    k = len(COUNT_NAMES)
    values = np.r_[record.values[:k], record.values[2 * k : 2 * k + 8], record.values[-1]] * h
    return np.r_[values, h], record.group_coverage_fraction


def _compose(raw):
    h = raw[-1]
    if h <= 0:
        raise ValueError("Nonpositive mean heavy-atom population")
    d = raw[:-1] / h
    k = len(COUNT_NAMES)
    if (d[:k] < -1e-10).any() or not np.isfinite(d).all():
        raise ValueError("Invalid block population moments")
    c = np.maximum(d[:k], 0)
    return np.r_[c, np.sqrt(c), d[k : k + 8], 1 / h, 1 / np.sqrt(h), max(d[-1], 0)]


def characterize_surfactant(record: SurfactantInput):
    if record.ionic or record.counterion_smiles is not None:
        raise ValueError("Ionic/counterion material needs a separate physical backend")
    dist = record.eo_po
    if dist.sequence != "block":
        raise ValueError("Random/specified/unknown sequences need explicit sequence information")
    if dist.eo_mean and dist.po_mean and dist.block_order is None:
        raise ValueError("Mixed EO/PO blocks require an explicit block_order")
    order = dist.block_order or "eo_then_po"
    for cap in (record.tail_smiles, record.head_smiles):
        mol = _fragment(cap, 1)
        if any(a.GetFormalCharge() for a in mol.GetAtoms()):
            raise ValueError("Formally charged caps are not a neutral surfactant profile")
    cache = {}

    def population(eo, po):
        key = (int(eo > 0), int(po > 0))
        if eo + po <= 64 and float(eo).is_integer() and float(po).is_integer():
            return _raw(
                assemble_blocks(record.tail_smiles, record.head_smiles, int(eo), int(po), order)
            )[0]
        if key not in cache:
            # Each positive block contributes a constant local environment beyond four units.
            a = np.array(
                _raw(
                    assemble_blocks(
                        record.tail_smiles, record.head_smiles, 4 * key[0], 4 * key[1], order
                    )
                )[0]
            )
            se = np.zeros_like(a)
            sp = np.zeros_like(a)
            if key[0]:
                se = (
                    _raw(
                        assemble_blocks(
                            record.tail_smiles, record.head_smiles, 16, 4 * key[1], order
                        )
                    )[0]
                    - a
                ) / 12
            if key[1]:
                sp = (
                    _raw(
                        assemble_blocks(
                            record.tail_smiles, record.head_smiles, 4 * key[0], 16, order
                        )
                    )[0]
                    - a
                ) / 12
            offset = a - se * (4 * key[0]) - sp * (4 * key[1])
            test = _raw(
                assemble_blocks(
                    record.tail_smiles, record.head_smiles, 32 * key[0], 32 * key[1], order
                )
            )[0]
            if not np.allclose(
                test, se * 32 * key[0] + sp * 32 * key[1] + offset, atol=1e-7, rtol=1e-9
            ):
                raise ValueError("Block descriptor additivity validation failed")
            k = len(COUNT_NAMES)
            for v in (se, sp, offset):
                v[:k] = np.rint(v[:k])
                v[-2:] = np.rint(v[-2:])  # unassigned and heavy-atom counts
            cache[key] = (se, sp, offset)
        se, sp, offset = cache[key]
        return eo * se + po * sp + offset

    issues = []
    if dist.po_mean:
        issues.append("po_regiochemistry_O_CH2_CHMe_tacticity_unspecified")
    if dist.distribution == "empirical":
        raw_mean = sum(w * population(e, p) for e, p, w in dist.joint_pmf)
        basis = "joint_pmf_expected_atom_and_descriptor_populations"
    else:
        raw_mean = population(dist.eo_mean, dist.po_mean)
        basis = "declared_count_means_moment_approximation"
        if dist.distribution != "monodisperse":
            issues.append(
                "means_do_not_specify_expected_nonlinear_features_or_zero_block_probability"
            )
    if not np.isfinite(raw_mean).all():
        raise ValueError("Nonfinite mean populations")
    # Mass is exact and additive for declared EO/PO counts, independent of block order.
    cap_smiles = assemble_blocks(record.tail_smiles, record.head_smiles, 0, 0)
    end_mass = Descriptors.MolWt(Chem.MolFromSmiles(cap_smiles))
    eo_mass = (
        Descriptors.MolWt(
            Chem.MolFromSmiles(assemble_blocks(record.tail_smiles, record.head_smiles, 1, 0))
        )
        - end_mass
    )
    po_mass = (
        Descriptors.MolWt(
            Chem.MolFromSmiles(assemble_blocks(record.tail_smiles, record.head_smiles, 0, 1))
        )
        - end_mass
    )
    calculated_mn = end_mass + dist.eo_mean * eo_mass + dist.po_mean * po_mass
    if record.mn_g_mol is not None and abs(calculated_mn - record.mn_g_mol) > max(
        0.01 * record.mn_g_mol, 0.1
    ):
        issues.append("declared_mn_and_eo_po_means_inconsistent")
    features = _compose(raw_mean)
    return {
        "series_id": record.series_id,
        "feature_version": INTENSIVE_VERSION,
        "descriptor_provenance": descriptor_provenance(),
        "features": features.tolist(),
        "population_basis": basis,
        "calculated_mn_g_mol": calculated_mn,
        "cap_mass_g_mol": end_mass,
        "eo_residue_mass_g_mol": eo_mass,
        "po_residue_mass_g_mol": po_mass,
        "issues": issues,
        "group_coverage_fraction": 1 - raw_mean[-2] / raw_mean[-1],
        "mean_heavy_atoms": raw_mean[-1],
        "block_order": order,
        "max_explicit_block_counts": 32,
        "hsp_predictions": None,
        "physical_accuracy_validated": False,
        "caveat": (
            "Moment features are not measured HSP or a full distribution/material-phase model."
        ),
    }
