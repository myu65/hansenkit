"""Read reported material observations; never predict or fit an HSP value."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from .provenance import file_hash
from .schema import StrictModel

HASH = r"^[0-9a-f]{64}$"
Number = StrictFloat | StrictInt


def _nonblank(value):
    if not value.strip():
        raise ValueError("Source identity and permission evidence must be nonblank")
    return value


class ReviewedSourceMember(StrictModel):
    row_count: StrictInt = Field(ge=1)
    material_ids: tuple[str, ...] = Field(min_length=1)


class ObservationReview(StrictModel):
    schema_version: Literal[1] = 1
    sha256: str = Field(pattern=HASH)
    source_asset_path: str
    source_asset_sha256: str = Field(pattern=HASH)
    preparation_review_path: str
    preparation_review_sha256: str = Field(pattern=HASH)
    source_members: dict[str, ReviewedSourceMember]
    material_ids: tuple[str, ...]
    source: str
    source_license: str
    permission_evidence: str
    approved_by: str
    rights_status: Literal["approved"]
    audit_basis: Literal["operator_assumption", "documented_permission"]
    use_scope: Literal["scientific_private_noncommercial"]
    allowed_role: Literal["evaluation_only"]
    read_only_lookup_allowed: Literal[True]
    training_allowed: Literal[False]
    derived_labels_allowed: Literal[False]
    derived_weights_allowed: Literal[False]
    hsp_inference_allowed: Literal[False]
    public_redistribution: Literal[False]

    @field_validator("source", "source_license", "permission_evidence", "approved_by")
    @classmethod
    def evidence(cls, value):
        return _nonblank(value)

    @field_validator("material_ids")
    @classmethod
    def unique_materials(cls, values):
        if not values or len(set(values)) != len(values):
            raise ValueError("Distinct reviewed material IDs are required")
        return tuple(_nonblank(value) for value in values)

    @field_validator("source_members")
    @classmethod
    def members(cls, values):
        if not values or any(not key.strip() for key in values):
            raise ValueError("Reviewed source members need positive row counts")
        return values

    @model_validator(mode="after")
    def member_material_support(self):
        member_ids = {
            value for member in self.source_members.values() for value in member.material_ids
        }
        if member_ids != set(self.material_ids):
            raise ValueError("Source member material IDs differ from reviewed materials")
        return self

    @field_validator(
        "read_only_lookup_allowed",
        "training_allowed",
        "derived_labels_allowed",
        "derived_weights_allowed",
        "hsp_inference_allowed",
        "public_redistribution",
        mode="before",
    )
    @classmethod
    def permissions(cls, value):
        if type(value) is not bool:
            raise ValueError("Permission flags require JSON booleans")
        return value


class ReportedWeight(StrictModel):
    kind: Literal["Mn", "Mw", "unspecified"]
    value_g_mol: Number | None = Field(default=None, gt=0)
    value_as_reported: str | None = None
    unit_as_reported: str | None = None
    method: str | None

    @field_validator("value_as_reported", "unit_as_reported")
    @classmethod
    def source_literals(cls, value):
        return _nonblank(value) if value is not None else None

    @model_validator(mode="after")
    def weight_evidence(self):
        if self.value_g_mol is None and self.value_as_reported is None:
            raise ValueError("A qualified g/mol value or a literal source report is required")
        return self


class MaterialObservation(StrictModel):
    candidate_id: str
    source_asset_sha256: str = Field(pattern=HASH)
    source_member: str
    source_row_index_zero_based: StrictInt = Field(ge=0)
    material_id: str
    polymer_abbreviation: str
    reported_molecular_weight: ReportedWeight
    mn_g_mol: Number | None = Field(gt=0)
    dispersity: Number | None = Field(ge=1)
    end_groups: tuple[str | None, str | None] | None
    repeat_graph_with_source_supported_ports: str | None
    numeric_temperature_k: Number | None = Field(gt=0)
    material_phase: str | None
    source_reported_concentration_wv_percent: Number | None = Field(gt=0, le=100)
    source_reported_polymer_mass_g: Number | None = Field(gt=0)
    source_reported_solvent_volume_ml: Number | None = Field(gt=0)
    observation_wait_h: Number | None = Field(gt=0)
    solvent_name_as_reported: str
    solvent_identity_smiles: str | None
    binary_observation: StrictInt = Field(ge=0, le=1)
    qualitative_note_as_reported: str | None
    positive_definition: str
    negative_definition: str
    label_kind: Literal[
        "experimental_compatibility_observation_candidate",
        "published_reference_compatibility_observation",
        "authored_observation_fixture",
    ]
    allowed_role: Literal["evaluation_candidate_only", "authored_fixture_only"]
    hsp_components: None
    hsp_predictions: None
    physical_accuracy_validated: Literal[False]

    @field_validator(
        "candidate_id",
        "source_member",
        "material_id",
        "polymer_abbreviation",
        "solvent_name_as_reported",
        "positive_definition",
        "negative_definition",
    )
    @classmethod
    def identities(cls, value):
        return _nonblank(value)

    @field_validator("physical_accuracy_validated", mode="before")
    @classmethod
    def no_qualification(cls, value):
        if type(value) is not bool:
            raise ValueError("Qualification requires a JSON boolean")
        return value

    @model_validator(mode="after")
    def authored_label_role(self):
        authored = self.label_kind == "authored_observation_fixture"
        if authored != (self.allowed_role == "authored_fixture_only"):
            raise ValueError("Authored fixtures and source observations need separate roles")
        return self


class ObservationLedger(StrictModel):
    scope: str
    audit_basis: Literal["operator_assumption", "documented_permission"]
    preparation_review_sha256: str = Field(pattern=HASH)
    records: tuple[MaterialObservation, ...] = Field(min_length=1)


class ObservationQuery(StrictModel):
    material_id: str
    solvent_name_as_reported: str | None = None

    @field_validator("material_id", "solvent_name_as_reported")
    @classmethod
    def exact_source_string(cls, value):
        return _nonblank(value) if value is not None else None


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Repeated JSON field is ambiguous")
        result[key] = value
    return result


def _decode(raw):
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs)


class SourceObservationLookup:
    """Exact source grade and solvent lookup; no condition or structural generalization."""

    def __init__(self, data_path, review_path):
        review_path = Path(review_path)
        review_raw = review_path.read_bytes()
        review = ObservationReview.model_validate(_decode(review_raw))
        # Check upstream files opaquely, without decoding their numerical tables or code.
        for name in ("source_asset", "preparation_review"):
            asset = Path(getattr(review, name + "_path"))
            if not asset.is_absolute():
                asset = review_path.parent / asset
            if file_hash(asset) != getattr(review, name + "_sha256"):
                raise ValueError("Reviewed source/preparation checksum mismatch")
        raw = Path(data_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != review.sha256:
            raise ValueError("Observation ledger checksum mismatch")
        ledger = ObservationLedger.model_validate(_decode(raw))
        if (
            ledger.preparation_review_sha256 != review.preparation_review_sha256
            or ledger.audit_basis != review.audit_basis
        ):
            raise ValueError("Observation preparation/audit provenance differs")
        ids, origins = set(), set()
        for record in ledger.records:
            origin = (record.source_member, record.source_row_index_zero_based)
            if record.candidate_id in ids or origin in origins:
                raise ValueError("Observation/source row IDs must be distinct")
            if (
                record.source_asset_sha256 != review.source_asset_sha256
                or record.material_id not in review.material_ids
                or record.source_member not in review.source_members
                or record.material_id
                not in review.source_members[record.source_member].material_ids
                or record.source_row_index_zero_based
                >= review.source_members[record.source_member].row_count
            ):
                raise ValueError("Observation row is outside reviewed source/material support")
            ids.add(record.candidate_id)
            origins.add(origin)
        self._records = ledger.records
        self._provenance = {
            "data_sha256": review.sha256,
            "review_sha256": hashlib.sha256(review_raw).hexdigest(),
            "source_asset_sha256": review.source_asset_sha256,
            "preparation_review_sha256": review.preparation_review_sha256,
            "source": review.source,
            "source_license": review.source_license,
            "audit_basis": review.audit_basis,
            "use_scope": review.use_scope,
            "original_ledger_scope": ledger.scope,
            "operation": "read_only_source_lookup",
            "training_allowed": False,
            "public_redistribution": False,
            "model_execution": False,
            "hsp_predictions": None,
            "physical_accuracy_validated": False,
        }

    def lookup(self, query):
        query = ObservationQuery.model_validate(query)
        rows = [
            record.model_dump(mode="json")
            for record in self._records
            if record.material_id == query.material_id
            and (
                query.solvent_name_as_reported is None
                or record.solvent_name_as_reported == query.solvent_name_as_reported
            )
        ]
        outcomes = {}
        for row in rows:
            outcomes.setdefault(row["solvent_name_as_reported"], set()).add(
                row["binary_observation"]
            )
        return {
            **self._provenance,
            "status": "reported_source_observation" if rows else "no_reported_observation",
            "query": query.model_dump(),
            "query_sha256": hashlib.sha256(query.model_dump_json().encode()).hexdigest(),
            "records": rows,
            "rows": len(rows),
            "same_query_key_outcome_disagreement": any(
                len(values) > 1 for values in outcomes.values()
            ),
            "unknown_conditions_are_not_matched_or_imputed": True,
            "grade_identity_is_a_source_assertion": True,
        }
