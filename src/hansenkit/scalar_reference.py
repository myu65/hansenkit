"""Approved-array scalar reference regression and original two-port unit descriptors.

Caller-reviewed arrays and pre-isolated families only; no source loader or HSP
backend. Group identifiers and provenance declarations do not prove rights or
chemical isolation. No units are converted and no reference value is decomposed.
"""

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import GroupKFold

from .charge import charge_classification
from .chemistry import RULES, queries

GRID = tuple((kind, alpha) for kind in ("ridge", "rbf") for alpha in (0.1, 1.0, 10.0))
ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53, 14, 5, 34)
FEATURE_NAMES = tuple(f"heavy_element_fraction_Z{z}" for z in ELEMENTS) + (
    "other_heavy_element_fraction",
    "real_heavy_atom_count",
    "aromatic_real_atom_fraction",
    "ring_real_atom_fraction",
    "raw_unit_HBD_per_real_heavy_atom",
    "raw_unit_HBA_per_real_heavy_atom",
    "raw_unit_TPSA_per_real_heavy_atom",
)
UNITS = {
    "single_delta_total": "source_declared_(J/cm3)^0.5",
    "CED": "source_declared_cal/cm3_cal_definition_unknown",
}
FALSE_FLAGS = (
    "public_redistribution_allowed",
    "commercial_allowed",
    "material_identity_qualified",
    "measurement_lineage_qualified",
    "physical_accuracy_qualified",
    "HSP_backend_qualified",
    "default_predictor_enabled",
    "calibrated_uncertainty_qualified",
)


class NumericalRefusal(ValueError):
    """The original numerical model is inadmissible; no fallback or repair occurs."""


def _require(ok, message):
    if not ok:
        raise NumericalRefusal(message)


def _provenance_check(provenance):
    _require(
        provenance["task"] in UNITS and provenance["source_unit"] == UNITS[provenance["task"]],
        "Exact separate source-unit task",
    )
    _require(
        provenance["label_kind"] in {"published_reference_auxiliary", "synthetic"}
        and provenance["audit_basis"] == "operator_assumption"
        and provenance["use_scope"] == "scientific_private_noncommercial",
        "Separate reference/synthetic private scope",
    )
    if provenance["label_kind"] == "synthetic":
        _require(
            provenance.get("authored_fixture_only") is True
            and provenance["allowed_role"] == "authored_fixture_only",
            "Synthetic own-fixture provenance remains separate",
        )
    else:
        _require(
            provenance.get("authored_fixture_only") is False
            and provenance["allowed_role"] == "source_specific_local_training",
            "Exact future source-training role",
        )
    _require(
        provenance["rights_status"] == "approved" and provenance["training_allowed"] is True,
        "Fresh caller-approved array fitting review",
    )
    _require(
        provenance.get("private_derived_weights_allowed") is True,
        "Explicit private derived-weight permission",
    )
    _require(
        all(provenance.get(key) is False for key in FALSE_FLAGS)
        and "HSP_predictions" in provenance
        and provenance["HSP_predictions"] is None,
        "No publication or physical qualification",
    )
    for key in (
        "source_asset_sha256",
        "whole_source_name_bridge_sha256",
        "reservation_sha256",
        "structure_split_sha256",
        "fitting_review_sha256",
    ):
        value = provenance[key]
        _require(
            type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "Exact declared provenance digests",
        )


def _families(values):
    """One nonempty string or nonboolean integer namespace, without coercion."""
    if isinstance(values, np.ndarray):
        _require(values.ndim == 1, "One scalar family key per supplied row")
        raw = values.tolist()
    else:
        _require(
            isinstance(values, (list, tuple)), "Family keys require a one-dimensional sequence"
        )
        raw = list(values)
    _require(raw, "Nonempty training family keys")
    kinds = set()
    for value in raw:
        if isinstance(value, str) and value.strip():
            kinds.add("string")
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            kinds.add("integer")
        else:
            raise NumericalRefusal("Family keys must be nonblank strings or nonboolean integers")
    _require(len(kinds) == 1, "No mixed integer/string family namespaces or implicit merging")
    return np.asarray(raw, dtype=object)


def _balanced_weights(groups):
    groups = _families(groups)
    _unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def _family_mse(truth, prediction, groups):
    groups = _families(groups)
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    _require(
        truth.shape == prediction.shape == groups.shape
        and truth.ndim == 1
        and np.isfinite(truth).all()
        and np.isfinite(prediction).all(),
        "Finite scalar original-unit metric",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sum(_balanced_weights(groups) * (prediction - truth) ** 2))
    _require(np.isfinite(value), "Finite original-unit family MSE")
    return value


def _gram(left, right, gamma):
    left, right = np.asarray(left, float), np.asarray(right, float)
    distance = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    _require(np.isfinite(distance).all(), "Finite original feature distance")
    return np.exp(-gamma * distance)


def _fit(x, y, groups, kind, alpha, provenance):
    _provenance_check(provenance)
    groups = _families(groups)
    x, y = np.asarray(x, float), np.asarray(y, float)
    _require(
        x.ndim == 2 and x.shape[1] == 19 and y.shape == (len(x),) and groups.shape == y.shape,
        "A19 scalar training arrays",
    )
    _require(
        np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all(),
        "Finite features and strict positive source labels; no epsilon",
    )
    _require(
        type(kind) is str and type(alpha) in (int, float) and (kind, alpha) in GRID,
        "Only six fixed heads; booleans are not hyperparameters",
    )
    weights = _balanced_weights(groups)
    with np.errstate(over="ignore", invalid="ignore"):
        center = x[0] + np.sum(weights[:, None] * (x - x[0]), axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        variance = np.sum(weights[:, None] * (x - center) ** 2, axis=0)
    _require(
        np.isfinite(variance).all() and (variance >= 0).all(), "Finite train-only feature variance"
    )
    mask = np.min(x, axis=0) != np.max(x, axis=0)
    _require((variance[mask] > 0).all(), "Nonconstant variance unresolved/underflow; no epsilon")
    _require(mask.any(), "No learned head without a varying training feature")
    scale = np.sqrt(variance[mask])
    z = (x[:, mask] - center[mask]) / scale
    log_y = np.log(y)
    mean = float(weights @ log_y)
    residual = log_y - mean
    state = {
        "schema_version": 1,
        "task": "scalar_reference_head",
        "kind": kind,
        "alpha": alpha,
        "feature_count": 19,
        "center": center.tolist(),
        "train_min": np.min(x, axis=0).tolist(),
        "train_max": np.max(x, axis=0).tolist(),
        "mask": mask.tolist(),
        "scale": scale.tolist(),
        "log_target_training_mean": mean,
        "weights": weights.tolist(),
        "runtime_numpy_version": np.__version__,
        "provenance": dict(provenance),
        "response": "exp_of_learned_logpositive_source_unit",
        "functional_channel_used_for_fitting": False,
    }
    sqrt_w = np.sqrt(weights)
    if kind == "ridge":
        matrix = z * sqrt_w[:, None]
        coefficient = np.linalg.solve(
            matrix.T @ matrix + alpha * np.eye(z.shape[1]), matrix.T @ (residual * sqrt_w)
        )
        state["coefficient"] = coefficient.tolist()
    else:
        gamma = 1.0 / 19.0
        kernel = _gram(z, z, gamma)
        row_mean = kernel @ weights
        global_mean = float(weights @ row_mean)
        centered = kernel - row_mean[:, None] - row_mean[None, :] + global_mean
        weighted = sqrt_w[:, None] * centered * sqrt_w[None, :]
        coefficient = sqrt_w * np.linalg.solve(weighted + alpha * np.eye(len(x)), residual * sqrt_w)
        state.update(
            gamma=gamma,
            training_coordinates=z.tolist(),
            kernel_training_row_mean=row_mean.tolist(),
            kernel_training_global_mean=global_mean,
            dual_coefficient=coefficient.tolist(),
        )
    _require(np.isfinite(coefficient).all(), "Finite trained coefficients")
    state["state_digest"] = _digest(state)
    _validate(state)
    return state


def _digest(state):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "state_digest"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _validate(state):
    _require(
        type(state["schema_version"]) is int
        and state["schema_version"] == 1
        and state["task"] == "scalar_reference_head"
        and type(state["kind"]) is str
        and type(state["alpha"]) in (int, float)
        and (state["kind"], state["alpha"]) in GRID
        and state["feature_count"] == 19,
        "Exact saved head schema/grid",
    )
    _provenance_check(state["provenance"])
    _require(
        state["runtime_numpy_version"] == np.__version__
        and state["response"] == "exp_of_learned_logpositive_source_unit"
        and state["functional_channel_used_for_fitting"] is False,
        "Exact runtime/response/scope on restore",
    )
    _require(state["state_digest"] == _digest(state), "Saved integrity digest, not authenticity")
    mask = np.asarray(state["mask"])
    _require(
        mask.shape == (19,) and mask.dtype == bool and mask.any(),
        "Boolean training-only feature mask",
    )
    center, scale, weights = (
        np.asarray(state[key], float) for key in ("center", "scale", "weights")
    )
    minimum, maximum = (np.asarray(state[key], float) for key in ("train_min", "train_max"))
    _require(
        minimum.shape == maximum.shape == (19,)
        and np.isfinite(minimum).all()
        and np.isfinite(maximum).all()
        and (minimum <= maximum).all()
        and (center >= minimum).all()
        and (center <= maximum).all(),
        "Saved exact finite training OOD bounds including inactive columns",
    )
    _require(
        center.shape == (19,)
        and scale.shape == (int(mask.sum()),)
        and weights.ndim == 1
        and len(weights) > 0,
        "Saved training state dimensions",
    )
    _require(
        np.isfinite(center).all()
        and np.isfinite(scale).all()
        and (scale > 0).all()
        and np.isfinite(weights).all()
        and (weights > 0).all()
        and abs(float(weights.sum()) - 1.0) <= 1e-12
        and np.isfinite(state["log_target_training_mean"]),
        "Saved finite train-only moments/weights",
    )
    if state["kind"] == "ridge":
        coefficient = np.asarray(state["coefficient"], float)
        _require(
            coefficient.shape == scale.shape and np.isfinite(coefficient).all(),
            "Saved Ridge coefficients",
        )
    else:
        coords = np.asarray(state["training_coordinates"], float)
        means, coefficient = (
            np.asarray(state[key], float)
            for key in ("kernel_training_row_mean", "dual_coefficient")
        )
        _require(
            coords.shape == (len(weights), int(mask.sum()))
            and means.shape == coefficient.shape == weights.shape
            and all(np.isfinite(value).all() for value in (coords, means, coefficient)),
            "Saved kernel dimensions/values",
        )
        _require(state["gamma"] == 1.0 / 19.0, "Fixed original feature-count bandwidth")
        matrix = _gram(coords, coords, state["gamma"])
        _require(
            np.allclose(matrix @ weights, means, rtol=1e-12, atol=1e-12)
            and abs(float(weights @ means) - state["kernel_training_global_mean"]) <= 1e-12,
            "Saved kernel-centering algebra",
        )


def _predict(state, x):
    _validate(state)
    x = np.asarray(x, float)
    _require(x.ndim == 2 and x.shape[1] == 19 and np.isfinite(x).all(), "Finite A19 query arrays")
    mask = np.asarray(state["mask"], bool)
    z = (x[:, mask] - np.asarray(state["center"])[mask]) / np.asarray(state["scale"])
    if state["kind"] == "ridge":
        correction = z @ np.asarray(state["coefficient"])
    else:
        kernel = _gram(z, np.asarray(state["training_coordinates"]), state["gamma"])
        centered = (
            kernel
            - (kernel @ np.asarray(state["weights"]))[:, None]
            - np.asarray(state["kernel_training_row_mean"])[None, :]
            + state["kernel_training_global_mean"]
        )
        correction = centered @ np.asarray(state["dual_coefficient"])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(state["log_target_training_mean"] + correction)
    _require(
        np.isfinite(result).all() and (result > 0).all(),
        "Finite positive response; no clipping or zero repair",
    )
    return result


def _ood(state, x):
    _validate(state)
    x = np.asarray(x, float)
    _require(
        x.ndim == 2 and x.shape[1] == 19 and np.isfinite(x).all(), "Finite A19 OOD query arrays"
    )
    outside = (x < np.asarray(state["train_min"])) | (x > np.asarray(state["train_max"]))
    return {
        "outside_training_range": outside.any(axis=1).tolist(),
        "outside_feature_indices": [np.flatnonzero(row).tolist() for row in outside],
        "scope": "descriptive_training_feature_range_including_inactive_columns",
        "physical_or_calibrated_uncertainty_qualified": False,
    }


def _select(x, y, groups, training_indices, provenance):
    _provenance_check(provenance)
    indices = np.asarray(training_indices)
    _require(
        indices.ndim == 1
        and indices.dtype.kind in "iu"
        and len(indices) == len(set(indices.tolist()))
        and len(indices) > 0,
        "Explicit unique training IDs",
    )
    _require(
        all(isinstance(values, (np.ndarray, list, tuple)) for values in (x, y, groups))
        and len(y) == len(groups) == len(x)
        and (indices >= 0).all()
        and (indices < len(x)).all(),
        "Training IDs within the caller-approved row domain",
    )

    # Select first: coercing an entire list/object array can invoke held-row protocols.
    def training_only(values):
        if isinstance(values, np.ndarray):
            return values[indices]
        return [values[int(index)] for index in indices]

    train_groups = _families(training_only(groups))
    train_x, train_y = (np.asarray(training_only(values)) for values in (x, y))
    _require(len(set(train_groups.tolist())) >= 5, "Five training families; no random fallback")
    folds = list(GroupKFold(n_splits=5).split(train_x, groups=train_groups))
    reports = []
    for kind, alpha in GRID:
        oof = np.full(len(indices), np.nan)
        errors = []
        for train, valid in folds:
            try:
                state = _fit(
                    train_x[train], train_y[train], train_groups[train], kind, alpha, provenance
                )
                oof[valid] = _predict(state, train_x[valid])
            except NumericalRefusal as error:
                errors.append(str(error))
        score = _family_mse(train_y, oof, train_groups) if not errors else None
        reports.append(
            {
                "kind": kind,
                "alpha": alpha,
                "family_macro_original_unit_MSE": score,
                "numerical_refusals": errors,
            }
        )
    valid = [
        (row["family_macro_original_unit_MSE"], i)
        for i, row in enumerate(reports)
        if row["family_macro_original_unit_MSE"] is not None
    ]
    _require(valid, "All six candidates refused; no guessed fallback")
    score, selected = min(valid)
    chosen = reports[selected]
    state = _fit(train_x, train_y, train_groups, chosen["kind"], chosen["alpha"], provenance)
    return state, {
        "selected_index": selected,
        "selection_family_macro_original_unit_MSE": score,
        "candidates": reports,
        "training_indices": indices.tolist(),
        "held_labels_or_features_used": False,
    }


@dataclass
class ScalarReferenceHead:
    """Positive scalar source-unit research head, never a qualified HSP backend.

    Input columns follow ``FEATURE_NAMES``. Callers approve data/derived weights
    and isolate canonical, name and polymer families before supplying arrays.
    Both source quantities have separate model provenance; CED calorie meaning
    remains unknown. A saved digest checks integrity, not authenticity, rights
    or original-target reconstruction. OOD is descriptive, not calibrated.
    """

    parameters: dict

    @classmethod
    def fit(cls, x, y, groups, *, kind, alpha, provenance):
        try:
            state = _fit(x, y, groups, kind, alpha, copy.deepcopy(provenance))
        except (KeyError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError) as error:
            raise NumericalRefusal(
                "Malformed or unsolved scalar-reference training arrays"
            ) from error
        return cls.restore(state)

    def state(self):
        return copy.deepcopy(self.parameters)

    @classmethod
    def restore(cls, state):
        try:
            state = copy.deepcopy(state)
            _validate(state)
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise NumericalRefusal("Malformed scalar-reference saved state") from error
        return cls(state)

    def predict(self, x):
        """Return positive research responses; refusals are never clipped/repaired."""
        try:
            return _predict(self.parameters, x)
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise NumericalRefusal("Malformed scalar-reference query or state") from error

    def ood(self, x):
        """Flag raw training-range departures in all19 columns, including constants."""
        try:
            return _ood(self.parameters, x)
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise NumericalRefusal("Malformed scalar-reference OOD query or state") from error


def fit_group_selected_scalar_reference(x, y, groups, *, training_indices, provenance):
    """Choose only six fixed heads inside five supplied training-family folds.

    Held rows do not enter target conversion, preprocessing or configuration
    selection. Every inner fold refits moments and its positive-response head.
    Caller group IDs do not establish chemical isolation; no split is made here.
    """
    try:
        state, report = _select(x, y, groups, training_indices, copy.deepcopy(provenance))
    except (KeyError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError) as error:
        raise NumericalRefusal("Malformed or unsolved scalar-reference selection inputs") from error
    return ScalarReferenceHead.restore(state), report


def raw_repeat_unit_features(smiles):
    """A19 of an unchanged, bounded raw two-port unit plus separate ownership audit.

    No canonicalization, chain assembly, caps, ends or molecular-mass inference
    occurs. HBD/HBA/TPSA are unit descriptors, not hydrogen-bond strength. Full
    elemental coverage never substitutes for an incomplete SMARTS vocabulary.
    """
    if type(smiles) is not str or not smiles or len(smiles) > 8192:
        raise ValueError("Bounded original source two-port string")
    with Chem.rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Original unit parse/connectedness; no repair")
    ports = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(ports) != 2 or any(
        p.GetDegree() != 1
        or p.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE
        or p.GetNeighbors()[0].GetAtomicNum() == 0
        for p in ports
    ):
        raise ValueError("Exactly two original singly attached ports; no caps")
    if charge_classification(mol) not in {
        "no_formal_charge",
        "neutral_covalent_nitro_or_nitrate",
    } or any(atom.GetNumRadicalElectrons() or atom.GetIsotope() for atom in mol.GetAtoms()):
        raise ValueError("Unreviewed charge/radical/isotope unit")
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy or mol.GetNumAtoms() > 512:
        raise ValueError("Bounded real-heavy unit")
    count = len(heavy)
    elements = [sum(atom.GetAtomicNum() == z for atom in heavy) for z in ELEMENTS]
    elements.append(sum(atom.GetAtomicNum() not in ELEMENTS for atom in heavy))
    values = np.asarray(
        [value / count for value in elements]
        + [
            float(count),
            sum(atom.GetIsAromatic() for atom in heavy) / count,
            sum(atom.IsInRing() for atom in heavy) / count,
            rdMolDescriptors.CalcNumHBD(mol) / count,
            rdMolDescriptors.CalcNumHBA(mol) / count,
            rdMolDescriptors.CalcTPSA(mol) / count,
        ]
    )
    if values.shape != (19,) or not np.isfinite(values).all() or sum(elements) != count:
        raise ValueError("Composition every real-heavy atom exactly once")
    owners, functional_counts = {}, {}
    for rule, query in zip(RULES, queries(), strict=True):
        if query is None:
            raise ValueError("Invalid original functional rule")
        functional_counts[rule.name] = 0
        for match in sorted(mol.GetSubstructMatches(query, uniquify=True, maxMatches=0)):
            owned = tuple(match[i] for i in rule.owned_query_atoms)
            if any(index in owners for index in owned):
                continue
            functional_counts[rule.name] += 1
            owners.update(dict.fromkeys(owned, rule.name))
    real_indices = {atom.GetIdx() for atom in heavy}
    if not set(owners) <= real_indices:
        raise ValueError("Functional ownership must exclude ports/hydrogen")
    missing = sorted(real_indices - set(owners))
    return values, {
        "original_source_SMILES_unchanged": smiles,
        "source_atom_index_namespace": "original_parsed_unit",
        "explicit_original_ports": [p.GetIdx() for p in ports],
        "real_heavy_composition_count": count,
        "composition_atom_coverage_fraction": 1.0,
        "composition_other_element_bucket_does_not_qualify_functional_coverage": True,
        "HBD_HBA_TPSA_proxy_scope": (
            "original_raw_two_port_unit_not_hbond_strength_or_assembled_chain"
        ),
        "functional_counts": functional_counts,
        "functional_owners": owners,
        "functional_unassigned_real_atoms": missing,
        "functional_coverage_fraction": len(owners) / count,
        "functional_atom_coverage_complete": not missing,
        "functional_channel_used_for_fitting": False,
        "caps_Mn_Mw_ends_or_grouping_key_feature_substitution": False,
        "physical_or_material_identity_qualified": False,
        "HSP_predictions": None,
    }
