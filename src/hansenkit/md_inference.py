"""Reviewed exact-source cached MD inference, separate from physical HSP prediction."""

import copy
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, field_validator
from rdkit import Chem
from rdkit.Chem import Descriptors

from .chains import assemble_homopolymer
from .chemistry import normalize_smiles
from .descriptors import descriptor_provenance
from .intensive import COUNT_NAMES, INTENSIVE_VERSION, intensive_features
from .schema import RepeatUnit, StrictModel

TARGETS = ("sp_vdw", "sp_ele")
RUNTIME_PACKAGES = ("numpy", "scikit-learn", "rdkit")
_HASH = r"^[0-9a-f]{64}$"


class MDInferenceReview(StrictModel):
    model_sha256: str = Field(pattern=_HASH)
    support_sha256: str = Field(pattern=_HASH)
    plan_sha256: str = Field(pattern=_HASH)
    fitting_review_sha256: str = Field(pattern=_HASH)
    reservation_sha256: str = Field(pattern=_HASH)
    fresh_selection_sha256: str = Field(pattern=_HASH)
    source_sha256: str = Field(pattern=_HASH)
    qualification_sha256: str = Field(pattern=_HASH)
    runtime_versions: dict[str, str]
    source: str = Field(min_length=1)
    declared_license: str = Field(min_length=1)
    permission_evidence: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    audit_basis: Literal["operator_assumption", "documented_permission"]
    use_scope: Literal["scientific_private_noncommercial"]
    rights_status: Literal["approved"]
    local_cached_inference_allowed: Literal[True]
    public_redistribution: Literal[False]
    encoder_execution_allowed: Literal[False]
    hsp_inference_allowed: Literal[False]

    @field_validator("source", "declared_license", "permission_evidence", "approved_by")
    @classmethod
    def nonblank_evidence(cls, value):
        if not value.strip():
            raise ValueError("Review evidence must be nonblank")
        return value

    @field_validator(
        "local_cached_inference_allowed",
        "public_redistribution",
        "encoder_execution_allowed",
        "hsp_inference_allowed",
        mode="before",
    )
    @classmethod
    def actual_boolean(cls, value):
        if type(value) is not bool:
            raise ValueError("Permission flags require JSON booleans")
        return value


class MDMaterialInput(StrictModel):
    source_uuid: str = Field(min_length=1)
    repeat_unit_smiles: str = Field(min_length=1, max_length=8192)
    dp: StrictInt = Field(ge=1)
    mn_g_mol: StrictFloat = Field(gt=0)
    dispersity: StrictFloat
    left_cap: str
    right_cap: str
    tacticity: Literal["none", "atactic"]
    architecture: Literal["linear"]
    temperature_k: StrictFloat = Field(gt=0)
    pressure_atm: StrictFloat = Field(gt=0)
    forcefield: str
    sp_module_version: str


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _numbers(value, shape):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Invalid numeric model state") from error
    _require(array.shape == shape and np.isfinite(array).all(), "Invalid numeric model state")
    array.setflags(write=False)
    return array


def _head(value, columns):
    _require(
        isinstance(value, dict)
        and set(value) == {"kind", "mean", "scale", "coef", "intercept"}
        and value["kind"] == "ridge",
        "Require a numeric Ridge head",
    )
    state = {
        name: _numbers(value[name], shape)
        for name, shape in (
            ("mean", (columns,)),
            ("scale", (columns,)),
            ("coef", (2, columns)),
            ("intercept", (2,)),
        )
    }
    _require((state["scale"] > 0).all(), "Model scales must be positive")
    return state


def _predict(head, values):
    return ((values - head["mean"]) / head["scale"]) @ head["coef"].T + head["intercept"]


class MDCohesionPredictor:
    """Infer only registered source states, from explicitly reviewed frozen arrays."""

    def __init__(self, model_path, review_path, support_path):
        review_path = Path(review_path)
        review = MDInferenceReview.model_validate_json(review_path.read_text(encoding="utf-8-sig"))
        _require(
            set(review.runtime_versions) == set(RUNTIME_PACKAGES)
            and all(version(p) == review.runtime_versions[p] for p in RUNTIME_PACKAGES),
            "Reviewed inference runtime differs",
        )
        _require(_hash(model_path) == review.model_sha256, "MD model checksum mismatch")
        _require(_hash(support_path) == review.support_sha256, "MD support checksum mismatch")
        model = json.loads(Path(model_path).read_text(encoding="utf-8-sig"))
        _require(isinstance(model, dict), "Require an MD model object")
        expected = {
            "artifact_type": "computed_md_cohesion_regression",
            "label_kind": "computed_md_cohesion",
            "target_names": list(TARGETS),
            "units": "MPa^0.5",
            "forcefield": "GAFF2_mod",
            "source_sp_module_version": "0.1.1",
            "use_scope": "scientific_private_noncommercial",
            "plan_sha256": review.plan_sha256,
            "fitting_review_sha256": review.fitting_review_sha256,
            "expanded_reservation_sha256": review.reservation_sha256,
            "fresh_selection_sha256": review.fresh_selection_sha256,
        }
        _require(all(model.get(k) == v for k, v in expected.items()), "MD model contract differs")
        _require(
            isinstance(model.get("audit_basis"), str)
            and model["audit_basis"] in {"operator_assumption", "documented_permission"},
            "Missing model audit basis",
        )
        for key in (
            "public_redistribution",
            "individual_polar_hydrogen_components_identified",
            "independent_experimental_accuracy_validated",
            "physical_hsp_backend_qualified",
        ):
            _require(
                model.get(key) is False, "MD model cannot acquire HSP or release qualification"
            )
        _require(
            "hsp_predictions" in model and model["hsp_predictions"] is None,
            "MD is not an HSP model",
        )
        _require(
            type(model.get("temperature_k")) in {int, float}
            and model["temperature_k"] == 300
            and type(model.get("pressure_atm")) in {int, float}
            and model["pressure_atm"] == 1,
            "Unsupported source conditions",
        )
        _require(
            isinstance(model.get("encoder_metadata"), dict)
            and model["encoder_metadata"].get("fixed") is True,
            "Missing fixed frozen encoder provenance",
        )
        models = model.get("models")
        _require(
            isinstance(models, dict) and set(models) == {"A", "B", "C"}, "Require A/B/C models"
        )
        _require(
            isinstance(models["C"], dict) and set(models["C"]) == {"chemical", "residual"},
            "Invalid residual model",
        )
        self._heads = {
            "A": _head(models["A"], 109),
            "B": _head(models["B"], 772),
            "C_chemical": _head(models["C"]["chemical"], 109),
            "C_residual": _head(models["C"]["residual"], 772),
        }
        _require(
            all(
                np.array_equal(self._heads["A"][k], self._heads["C_chemical"][k])
                for k in self._heads["A"]
            ),
            "Residual model must retain its final chemical head",
        )
        self._lower = _numbers(model.get("training_feature_min"), (109,))
        self._upper = _numbers(model.get("training_feature_max"), (109,))
        _require((self._lower <= self._upper).all(), "Invalid training feature bounds")
        _numbers(model.get("training_only_weighted_mean"), (2,))
        calibration = model.get("calibration")
        _require(
            isinstance(calibration, dict) and set(calibration) == {"A", "B", "C"},
            "Missing computed-reference calibration",
        )
        self._radii = {}
        for mode, state in calibration.items():
            _require(
                isinstance(state, dict) and state.get("computed_reference_only") is True,
                "Calibration is computed-reference only",
            )
            _require(state.get("nominal_coverage") == 0.9, "Unsupported calibration level")
            count = state.get("family_score_count")
            order = state.get("order_statistic_1based")
            _require(
                type(count) is int
                and count > 0
                and type(order) is int
                and order == math.ceil((count + 1) * 0.9) <= count,
                "Invalid calibration family rank",
            )
            scale = _numbers(state.get("target_scale"), (2,))
            radii = _numbers(state.get("interval_half_widths"), (2,))
            multiplier = state.get("multiplier")
            _require(
                type(multiplier) in {int, float}
                and math.isfinite(multiplier)
                and multiplier >= 0
                and (scale > 0).all()
                and (radii >= 0).all(),
                "Invalid computed calibration radius",
            )
            _require(
                np.allclose(radii, scale * multiplier, rtol=1e-12, atol=1e-12),
                "Calibration scales and radius differ",
            )
            self._radii[mode] = radii
        with np.load(support_path, allow_pickle=False) as archive:
            keys = {
                "source_uuid",
                "repeat_smiles",
                "capped_smiles",
                "dp",
                "mn_g_mol",
                "tacticity",
                "features",
                "embeddings",
                "metadata_json",
            }
            _require(set(archive.files) == keys, "Invalid source support columns")
            self._support = {k: archive[k].copy() for k in keys}
        support = self._support
        _require(support["source_uuid"].ndim == 1, "Support UUID must be a vector")
        n = len(support["source_uuid"])
        _require(n > 0, "Empty support registry")
        for key in ("source_uuid", "repeat_smiles", "capped_smiles", "tacticity"):
            values = support[key]
            _require(
                values.shape == (n,)
                and values.dtype.kind == "U"
                and all(str(v).strip() for v in values),
                "Invalid support identity column",
            )
        _require(len(set(support["source_uuid"].tolist())) == n, "Ambiguous source UUID")
        _require(
            support["dp"].shape == (n,)
            and support["dp"].dtype.kind in "iu"
            and (support["dp"] > 0).all(),
            "Invalid support DP",
        )
        for key, shape in (("mn_g_mol", (n,)), ("features", (n, 109)), ("embeddings", (n, 768))):
            _require(
                support[key].shape == shape
                and support[key].dtype.kind == "f"
                and np.isfinite(support[key]).all(),
                "Invalid support numeric array",
            )
        _require(
            (support["mn_g_mol"] > 0).all() and set(support["tacticity"]) <= {"none", "atactic"},
            "Invalid source material state",
        )
        raw = support["metadata_json"]
        _require(
            raw.shape == () and raw.dtype.kind == "U", "Support metadata must be scalar JSON text"
        )
        metadata = json.loads(str(raw.item()))
        _require(isinstance(metadata, dict), "Support metadata must be an object")
        expected_metadata = {
            "encoder_metadata": model["encoder_metadata"],
            "source_sha256": review.source_sha256,
            "qualification_sha256": review.qualification_sha256,
            "reservation_sha256": review.reservation_sha256,
            "selection_sha256": review.fresh_selection_sha256,
        }
        _require(
            all(metadata.get(k) == v for k, v in expected_metadata.items()),
            "Source support provenance differs",
        )
        _require(
            metadata.get("feature_version") == INTENSIVE_VERSION,
            "Source feature version differs",
        )
        _require(
            metadata.get("descriptor_provenance") == descriptor_provenance(),
            "Source descriptor parameters differ",
        )
        for value in support.values():
            value.setflags(write=False)
        self._indices = {str(v): i for i, v in enumerate(support["source_uuid"])}
        self._provenance = {
            **review.model_dump(),
            "review_sha256": _hash(review_path),
            "source_encoder_metadata": copy.deepcopy(model["encoder_metadata"]),
            "source_model_audit_basis": model["audit_basis"],
            "inference_provider": "hash-bound-source-registry-frozen-vectors",
            "feature_version": INTENSIVE_VERSION,
            "descriptor_provenance": descriptor_provenance(),
        }

    def _report(self, mode, reason=None):
        return {
            "status": "refused" if reason else "computed_reference_prediction",
            "mode": mode,
            "reasons": [reason] if reason else [],
            "predictions": None,
            "prediction_intervals": None,
            "label_kind": "computed_md_cohesion",
            "value_kind": "computed_reference",
            "units": "MPa^0.5",
            "temperature_k": 300,
            "pressure_atm": 1,
            "forcefield": "GAFF2_mod",
            "sp_module_version": "0.1.1",
            "use_scope": "scientific_private_noncommercial",
            "hsp_predictions": None,
            "individual_polar_hydrogen_components_identified": False,
            "independent_experimental_accuracy_validated": False,
            "physical_hsp_backend_qualified": False,
            "encoder_execution_performed": False,
            "provenance": copy.deepcopy(self._provenance),
        }

    def predict(self, record: dict, mode="A"):
        if mode not in ("A", "B", "C"):
            return self._report(mode, "unsupported_model_mode")
        try:
            request = MDMaterialInput.model_validate(record)
        except ValueError:
            return self._report(mode, "invalid_material_input")
        if (
            request.temperature_k,
            request.pressure_atm,
            request.forcefield,
            request.sp_module_version,
        ) != (300, 1, "GAFF2_mod", "0.1.1"):
            return self._report(mode, "unsupported_source_protocol")
        if request.dispersity != 1:
            return self._report(mode, "unsupported_molecular_weight_distribution")
        if request.source_uuid not in self._indices:
            return self._report(mode, "unregistered_source_material")
        row = self._indices[request.source_uuid]
        support = self._support
        try:
            unit = RepeatUnit(smiles=request.repeat_unit_smiles, mole_fraction=1).smiles
            mol = Chem.MolFromSmiles(unit)
            if any(
                a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
                for a in mol.GetAtoms()
            ):
                return self._report(mode, "unsupported_repeat_chemistry")
            if (
                normalize_smiles(request.left_cap) != "*C"
                or normalize_smiles(request.right_cap) != "*C"
            ):
                return self._report(mode, "unsupported_end_groups")
            expected_repeat = normalize_smiles(str(support["repeat_smiles"][row]))
            if (
                unit != expected_repeat
                or request.dp != int(support["dp"][row])
                or request.tacticity != support["tacticity"][row]
                or abs(request.mn_g_mol - support["mn_g_mol"][row]) > 0.05
            ):
                return self._report(mode, "registered_material_state_mismatch")
            first = assemble_homopolymer(unit, 1, "*C", "*C")
            second = assemble_homopolymer(unit, 2, "*C", "*C")
            first_mol = Chem.MolFromSmiles(first)
            if first_mol.GetNumHeavyAtoms() > 512 or first != normalize_smiles(
                str(support["capped_smiles"][row])
            ):
                return self._report(mode, "source_assembly_mismatch")
            m1 = Descriptors.MolWt(first_mol)
            residue_mass = Descriptors.MolWt(Chem.MolFromSmiles(second)) - m1
            derived_mass = m1 + (request.dp - 1) * residue_mass
            if (
                residue_mass <= 0
                or not math.isfinite(derived_mass)
                or abs(derived_mass - request.mn_g_mol) > 0.05
                or abs(derived_mass - support["mn_g_mol"][row]) > 0.05
            ):
                return self._report(mode, "source_mass_mismatch")
            intensive = intensive_features(first)
            heavy = first_mol.GetNumHeavyAtoms()
            chemical = np.r_[intensive.values, intensive.values[: len(COUNT_NAMES)] * heavy, heavy]
            rebuilt = np.r_[
                chemical,
                math.log1p(request.dp),
                math.log1p(derived_mass),
                1 / request.dp,
                float(request.tacticity == "atactic"),
            ]
            if not np.allclose(rebuilt, support["features"][row], rtol=1e-10, atol=1e-10):
                return self._report(mode, "source_feature_reconstruction_mismatch")
        except (ValueError, OverflowError):
            return self._report(mode, "source_structure_reconstruction_failed")
        x = support["features"][row]
        if ((x < self._lower) | (x > self._upper)).any():
            return self._report(mode, "outside_training_feature_bounds")
        z = np.r_[support["embeddings"][row], x[-4:]]
        with np.errstate(over="ignore", invalid="ignore"):
            if mode == "C":
                prediction = _predict(self._heads["C_chemical"], x) + _predict(
                    self._heads["C_residual"], z
                )
            else:
                prediction = _predict(self._heads[mode], x if mode == "A" else z)
        if not np.isfinite(prediction).all():
            return self._report(mode, "nonfinite_computed_prediction")
        if (prediction < 0).any():
            return self._report(mode, "negative_computed_prediction")
        radius = self._radii[mode]
        with np.errstate(over="ignore", invalid="ignore"):
            interval = np.c_[prediction - radius, prediction + radius]
        if not np.isfinite(interval).all():
            return self._report(mode, "nonfinite_computed_interval")
        result = self._report(mode)
        result["predictions"] = dict(zip(TARGETS, prediction.tolist(), strict=True))
        result["prediction_intervals"] = {
            name: {"lower": float(bounds[0]), "upper": float(bounds[1])}
            for name, bounds in zip(TARGETS, interval, strict=True)
        }
        result["interval_metadata"] = {
            "nominal_coverage": 0.9,
            "computed_reference_only": True,
            "physical_accuracy_or_exchangeability_qualified": False,
        }
        result["group_coverage_fraction"] = intensive.group_coverage_fraction
        return result
