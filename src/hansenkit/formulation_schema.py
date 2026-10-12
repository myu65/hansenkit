"""Source-grade/formulation observations; no concentration conversion or inferred endpoints."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

HASH = r"^[0-9a-f]{64}$"
Number = StrictFloat | StrictInt
Qualifier = Literal["as_reported", "approximately", "nominal", "unreported"]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )


def nonblank(value):
    if not value.strip():
        raise ValueError("Source identities/evidence must be nonblank")
    return value


class SourceCell(StrictModel):
    sheet_name: str
    excel_row_one_based: StrictInt = Field(ge=1)
    excel_column_one_based: StrictInt = Field(ge=1)

    @field_validator("sheet_name")
    @classmethod
    def original_name(cls, value):
        return nonblank(value)


class ReportedQuantity(StrictModel):
    value: Number | None
    unit: str | None
    unit_as_reported: str | None
    qualifier: Qualifier
    literal_as_reported: str | None
    source_cell: SourceCell | None

    @model_validator(mode="after")
    def unknown_is_explicit(self):
        if self.value is None and self.qualifier != "unreported":
            raise ValueError("Missing numeric values must remain unreported")
        if self.value is not None and (self.qualifier == "unreported" or not self.unit):
            raise ValueError("Reported values require original unit and qualifier")
        return self


class ReportedMass(StrictModel):
    value_g_mol: Number = Field(gt=0)
    statistic: Literal["Mn", "Mw", "unspecified"]
    qualifier: Qualifier
    literal_as_reported: str
    method_as_reported: str | None

    @model_validator(mode="after")
    def reported_source_mass(self):
        if self.qualifier == "unreported" or not self.literal_as_reported.strip():
            raise ValueError("Reported mass requires its original nominal/source literal")
        return self


class SourceGrade(StrictModel):
    source_grade_id: str
    name_as_reported: str
    nominal_eo_left: StrictInt | None = Field(ge=0)
    nominal_po: StrictInt | None = Field(ge=0)
    nominal_eo_right: StrictInt | None = Field(ge=0)
    block_counts_are_exact_DP: Literal[False]
    reported_mass: ReportedMass | None
    mn_g_mol: Number | None = Field(gt=0)
    dispersity: Number | None = Field(ge=1)
    end_groups_as_reported: tuple[str | None, str | None] | None
    source_supported_repeat_graph: str | None
    exact_material_structure_qualified: Literal[False]

    @field_validator(
        "block_counts_are_exact_DP", "exact_material_structure_qualified", mode="before"
    )
    @classmethod
    def no_numeric_bool(cls, value):
        if type(value) is not bool:
            raise ValueError("Source grade qualification requires JSON booleans")
        return value

    @field_validator("source_grade_id", "name_as_reported")
    @classmethod
    def names(cls, value):
        return nonblank(value)

    @model_validator(mode="after")
    def unspecified_mass_never_becomes_Mn(self):
        if self.reported_mass is not None and self.reported_mass.statistic != "Mn":
            if self.mn_g_mol is not None:
                raise ValueError("Unspecified/Mw source mass cannot supply Mn")
        return self


class FormulationConditions(StrictModel):
    polymer_wt_percent: ReportedQuantity
    salt_name_as_reported: str | None
    salt_mol_l: ReportedQuantity
    silica_wt_percent: ReportedQuantity
    silica_grade_as_reported: str | None
    silica_charge_as_reported: str | None
    silica_protonation_as_reported: str | None
    temperature_c: ReportedQuantity
    pH: ReportedQuantity
    pressure_pa: ReportedQuantity
    observation_wait_h: ReportedQuantity
    solvent_name_as_reported: str | None
    material_phase_as_reported: str | None
    polymer_present: StrictBool | None

    @model_validator(mode="after")
    def original_units_and_polymer_presence(self):
        units = {
            "polymer_wt_percent": "wt%",
            "salt_mol_l": "mol/L",
            "silica_wt_percent": "wt%",
            "temperature_c": "degC",
            "pH": "pH",
            "pressure_pa": "Pa",
            "observation_wait_h": "h",
        }
        for field, unit in units.items():
            quantity = getattr(self, field)
            if quantity.value is not None and quantity.unit != unit:
                raise ValueError("Original formulation units required; no conversions")
            if (
                field not in ("temperature_c", "pH")
                and quantity.value is not None
                and quantity.value < 0
            ):
                raise ValueError("Negative formulation quantity")
        for field in ("polymer_wt_percent", "silica_wt_percent"):
            value = getattr(self, field).value
            if value is not None and value > 100:
                raise ValueError("Source mass percentage outside 0..100")
        value = self.polymer_wt_percent.value
        expected = None if value is None else value > 0
        if self.polymer_present is not expected:
            raise ValueError("Zero-polymer controls are not polymer-present formulations")
        return self


class EndpointSupport(StrictModel):
    name_as_reported: str
    definition_as_reported: str | None
    literal_categories: tuple[str, ...] = Field(min_length=1)
    source_outcome_cells: tuple[SourceCell, ...] = Field(min_length=1)

    @field_validator("name_as_reported")
    @classmethod
    def name(cls, value):
        return nonblank(value)

    @field_validator("literal_categories")
    @classmethod
    def categories(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("Distinct literal source categories required")
        return tuple(nonblank(value) for value in values)


class EndpointObservation(StrictModel):
    candidate_id: str
    formulation_id: str
    source_asset_sha256: str = Field(pattern=HASH)
    source_outcome_cell: SourceCell
    source_title_cell: SourceCell | None
    source_axis_cells: tuple[SourceCell, ...]
    source_xf_index: StrictInt | None = Field(ge=0)
    source_formula_origin: str | None
    source_grade_id: str
    conditions: FormulationConditions
    endpoint_id: str
    endpoint_name_as_reported: str
    definition_as_reported: str | None
    literal_category_as_reported: str
    numerical_endpoint_value: None
    interpreted_binary_solubility: None
    label_kind: Literal["source_reported_formulation_observation", "authored_formulation_fixture"]
    allowed_role: Literal["evaluation_only", "authored_fixture_only"]
    hsp_components: None
    hsp_predictions: None
    physical_accuracy_validated: Literal[False]

    @field_validator("physical_accuracy_validated", mode="before")
    @classmethod
    def no_numeric_bool(cls, value):
        if type(value) is not bool:
            raise ValueError("Observation qualification requires JSON booleans")
        return value

    @field_validator(
        "candidate_id",
        "formulation_id",
        "source_grade_id",
        "endpoint_id",
        "endpoint_name_as_reported",
        "literal_category_as_reported",
    )
    @classmethod
    def identities(cls, value):
        return nonblank(value)

    @model_validator(mode="after")
    def separate_fixture_role(self):
        if (self.label_kind == "authored_formulation_fixture") != (
            self.allowed_role == "authored_fixture_only"
        ):
            raise ValueError("Authored fixtures and source observations need separate roles")
        return self


class SourceSheet(StrictModel):
    row_count: StrictInt = Field(ge=1)
    column_count: StrictInt = Field(ge=1)


class FormulationReview(StrictModel):
    schema_version: Literal[1] = 1
    ledger_sha256: str = Field(pattern=HASH)
    source_asset_path: str
    source_asset_sha256: str = Field(pattern=HASH)
    preparation_receipt_path: str
    preparation_receipt_sha256: str = Field(pattern=HASH)
    classification_receipt_path: str
    classification_receipt_sha256: str = Field(pattern=HASH)
    source_sheets: dict[str, SourceSheet]
    source_grades: dict[str, SourceGrade]
    endpoints: dict[str, EndpointSupport]
    expected_outcome_records: StrictInt = Field(ge=1)
    expected_formulations: StrictInt = Field(ge=1)
    expected_polymer_present_formulations: StrictInt = Field(ge=0)
    expected_zero_polymer_controls: StrictInt = Field(ge=0)
    expected_unknown_polymer_presence: StrictInt = Field(ge=0)
    source: str
    declared_license: str
    permission_evidence: str
    approved_by: str
    rights_status: Literal["approved"]
    audit_basis: Literal["operator_assumption", "documented_permission"]
    use_scope: Literal["scientific_private_noncommercial"]
    allowed_role: Literal["evaluation_only", "authored_fixture_only"]
    read_only_lookup_allowed: Literal[True]
    training_allowed: Literal[False]
    derived_labels_allowed: Literal[False]
    derived_weights_allowed: Literal[False]
    hsp_inference_allowed: Literal[False]
    physical_accuracy_validated: Literal[False]
    public_redistribution: Literal[False]

    @field_validator("source", "declared_license", "permission_evidence", "approved_by")
    @classmethod
    def evidence(cls, value):
        return nonblank(value)

    @field_validator(
        "read_only_lookup_allowed",
        "training_allowed",
        "derived_labels_allowed",
        "derived_weights_allowed",
        "hsp_inference_allowed",
        "physical_accuracy_validated",
        "public_redistribution",
        mode="before",
    )
    @classmethod
    def permissions_are_JSON_booleans(cls, value):
        if type(value) is not bool:
            raise ValueError("Rights/qualification flags require JSON booleans")
        return value

    @model_validator(mode="after")
    def support_and_population(self):
        if not all((self.source_sheets, self.source_grades, self.endpoints)):
            raise ValueError("Exact source sheets/grades/endpoints required")
        for mapping in (self.source_sheets, self.source_grades, self.endpoints):
            if any(not key.strip() for key in mapping):
                raise ValueError("Distinct nonblank support identities required")
        if any(key != grade.source_grade_id for key, grade in self.source_grades.items()):
            raise ValueError("Reviewed grade map differs from original grade identity")
        if self.expected_formulations != (
            self.expected_polymer_present_formulations
            + self.expected_zero_polymer_controls
            + self.expected_unknown_polymer_presence
        ):
            raise ValueError("All original formulations/controls/unknowns must be accounted for")
        return self


class FormulationLedger(StrictModel):
    schema_version: Literal[1] = 1
    scope: str
    audit_basis: Literal["operator_assumption", "documented_permission"]
    use_scope: Literal["scientific_private_noncommercial"]
    allowed_role: Literal["evaluation_only", "authored_fixture_only"]
    source_asset_sha256: str = Field(pattern=HASH)
    preparation_receipt_sha256: str = Field(pattern=HASH)
    classification_receipt_sha256: str = Field(pattern=HASH)
    source_grades: dict[str, SourceGrade]
    source_metadata_claims: tuple["SourceMetadataClaim", ...]
    records: tuple[EndpointObservation, ...] = Field(min_length=1)


class SourceMetadataClaim(StrictModel):
    literal_as_reported: str
    source_pointer: str
    interpretation: Literal["unqualified_source_metadata_claim"]

    @field_validator("literal_as_reported", "source_pointer")
    @classmethod
    def original_claim(cls, value):
        return nonblank(value)


class QuantityQuery(StrictModel):
    value: Number | None
    unit: str | None
    unit_as_reported: str | None = None
    qualifier: Qualifier


class ConditionQuery(StrictModel):
    polymer_wt_percent: QuantityQuery | None = None
    salt_name_as_reported: str | None = None
    salt_mol_l: QuantityQuery | None = None
    silica_wt_percent: QuantityQuery | None = None
    silica_grade_as_reported: str | None = None
    silica_charge_as_reported: str | None = None
    silica_protonation_as_reported: str | None = None
    temperature_c: QuantityQuery | None = None
    pH: QuantityQuery | None = None
    pressure_pa: QuantityQuery | None = None
    observation_wait_h: QuantityQuery | None = None
    solvent_name_as_reported: str | None = None
    material_phase_as_reported: str | None = None
    polymer_present: StrictBool | None = None


class FormulationQuery(StrictModel):
    source_grade_id: str
    endpoint_id: str | None = None
    conditions: ConditionQuery = Field(default_factory=ConditionQuery)

    @field_validator("source_grade_id", "endpoint_id")
    @classmethod
    def exact_source_strings(cls, value):
        return nonblank(value) if value is not None else None
