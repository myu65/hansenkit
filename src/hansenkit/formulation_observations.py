"""Read reviewed local endpoint records verbatim; retain duplicates, controls and conflicts."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from .formulation_schema import FormulationLedger, FormulationQuery, FormulationReview


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated JSON field is ambiguous")
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cell_identity(cell):
    return cell.sheet_name, cell.excel_row_one_based, cell.excel_column_one_based


def condition_key(condition):
    # Full source values/units/qualifiers and missingness matter, not evidence cell location.
    value = condition.model_dump(mode="json")
    for quantity in value.values():
        if isinstance(quantity, dict):
            quantity.pop("source_cell", None)
            quantity.pop("literal_as_reported", None)
            quantity.pop("unit_as_reported", None)
            if quantity.get("value") is not None:
                number = Decimal(str(quantity["value"]))
                require(number.is_finite(), "Grouping value must remain finite")
                if number.is_zero():
                    quantity["value"] = "0"
                else:
                    sign, digits, exponent = number.as_tuple()
                    digits = list(digits)
                    while digits[-1] == 0:
                        digits.pop()
                        exponent += 1
                    # No Decimal context rounding: preserve every finite original decimal digit.
                    quantity["value"] = (
                        ("-" if sign else "") + "".join(map(str, digits)) + "e" + str(exponent)
                    )
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class FormulationObservationLookup:
    def __init__(self, data_path, review_path):
        review_path = Path(review_path)
        review_raw = review_path.read_bytes()
        review = FormulationReview.model_validate(decode(review_raw))
        # Fresh lookup permission is checked before source bytes or candidate decode.
        for prefix in ("source_asset", "preparation_receipt", "classification_receipt"):
            path = Path(getattr(review, prefix + "_path"))
            if not path.is_absolute():
                path = review_path.parent / path
            require(
                sha(path) == getattr(review, prefix + "_sha256"),
                "Source/preparation checksum mismatch",
            )
        raw = Path(data_path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == review.ledger_sha256, "Ledger checksum mismatch")
        ledger = FormulationLedger.model_validate(decode(raw))
        require(
            ledger.source_asset_sha256 == review.source_asset_sha256
            and ledger.preparation_receipt_sha256 == review.preparation_receipt_sha256
            and ledger.classification_receipt_sha256 == review.classification_receipt_sha256
            and ledger.audit_basis == review.audit_basis
            and ledger.use_scope == review.use_scope
            and ledger.allowed_role == review.allowed_role
            and ledger.source_grades == review.source_grades,
            "Exact ledger asset/preparation/classification/grade/role provenance differs",
        )
        ids, origins, formulations = set(), set(), {}
        expected_origins = {
            (review.source_asset_sha256, *cell_identity(cell))
            for endpoint in review.endpoints.values()
            for cell in endpoint.source_outcome_cells
        }
        require(
            len(expected_origins)
            == sum(len(e.source_outcome_cells) for e in review.endpoints.values())
            == review.expected_outcome_records,
            "Exact endpoint cell allowlists cannot overlap or silently omit original outcomes",
        )

        def check_cell(cell):
            member = review.source_sheets.get(cell.sheet_name)
            require(
                member is not None
                and cell.excel_row_one_based <= member.row_count
                and cell.excel_column_one_based <= member.column_count,
                "Source cell outside exact reviewed sheet bounds",
            )

        for record in ledger.records:
            origin = record.source_asset_sha256, *cell_identity(record.source_outcome_cell)
            require(
                record.candidate_id not in ids and origin not in origins,
                "Distinct endpoint cell identities required",
            )
            endpoint = review.endpoints.get(record.endpoint_id)
            require(
                record.source_asset_sha256 == review.source_asset_sha256
                and record.source_grade_id in review.source_grades
                and endpoint is not None
                and record.endpoint_name_as_reported == endpoint.name_as_reported
                and record.definition_as_reported == endpoint.definition_as_reported
                and record.literal_category_as_reported in endpoint.literal_categories,
                "Record outside exact reviewed source grade/endpoint/category support",
            )
            require(
                record.source_outcome_cell in endpoint.source_outcome_cells,
                "Endpoint/cell pairing differs from source review",
            )
            require(
                record.allowed_role == ledger.allowed_role,
                "Source versus authored fixture role differs",
            )
            cells = [record.source_outcome_cell, *record.source_axis_cells]
            if record.source_title_cell is not None:
                cells.append(record.source_title_cell)
            cells += [
                q.source_cell
                for q in record.conditions.__dict__.values()
                if hasattr(q, "source_cell") and q.source_cell is not None
            ]
            for cell in cells:
                check_cell(cell)
            key = record.source_grade_id, condition_key(record.conditions)
            require(
                record.formulation_id not in formulations
                or formulations[record.formulation_id][0] == key,
                "Paired formulation ID cannot collapse different grades/conditions",
            )
            formulations[record.formulation_id] = key, record.conditions.polymer_present
            ids.add(record.candidate_id)
            origins.add(origin)
        counts = {
            value: sum(item[1] is value for item in formulations.values())
            for value in (True, False, None)
        }
        require(
            len(ledger.records) == review.expected_outcome_records
            and origins == expected_origins
            and len(formulations) == review.expected_formulations
            and counts[True] == review.expected_polymer_present_formulations
            and counts[False] == review.expected_zero_polymer_controls
            and counts[None] == review.expected_unknown_polymer_presence,
            "Complete original endpoint/formulation/control population required; no filtering",
        )
        self._records = ledger.records
        self._grades = ledger.source_grades
        self._metadata_claims = ledger.source_metadata_claims
        self._provenance = {
            "ledger_sha256": review.ledger_sha256,
            "review_sha256": hashlib.sha256(review_raw).hexdigest(),
            "source_asset_sha256": review.source_asset_sha256,
            "preparation_receipt_sha256": review.preparation_receipt_sha256,
            "classification_receipt_sha256": review.classification_receipt_sha256,
            "source": review.source,
            "declared_license": review.declared_license,
            "permission_evidence": review.permission_evidence,
            "audit_basis": review.audit_basis,
            "use_scope": review.use_scope,
            "allowed_role": review.allowed_role,
            "operation": "read_only_reported_formulation_lookup",
            "training_allowed": False,
            "derived_labels_allowed": False,
            "public_redistribution": False,
            "hsp_predictions": None,
            "physical_accuracy_validated": False,
            "model_execution": False,
            "original_ledger_scope": ledger.scope,
        }

    def lookup(self, query):
        query = FormulationQuery.model_validate(query)

        def matches(record):
            if record.source_grade_id != query.source_grade_id or (
                query.endpoint_id is not None and record.endpoint_id != query.endpoint_id
            ):
                return False
            for name in query.conditions.model_fields_set:
                value = getattr(query.conditions, name)
                actual = getattr(record.conditions, name)
                if hasattr(actual, "qualifier"):
                    if value is None:
                        if actual.value is not None:
                            return False
                    elif any(
                        getattr(actual, key) != getattr(value, key)
                        for key in (
                            "value",
                            "unit",
                            "qualifier",
                        )
                    ):
                        return False
                    elif "unit_as_reported" in value.model_fields_set and (
                        actual.unit_as_reported != value.unit_as_reported
                    ):
                        return False
                elif actual != value:
                    return False
            return True

        records = [record for record in self._records if matches(record)]
        outcomes = {}
        for record in records:
            key = record.source_grade_id, condition_key(record.conditions), record.endpoint_id
            outcomes.setdefault(key, set()).add(record.literal_category_as_reported)
        request = query.model_dump(mode="json", exclude_unset=True)
        return {
            **self._provenance,
            "status": "reported_formulation_observations" if records else "no_reported_observation",
            "query": request,
            "query_sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest(),
            "source_grade": self._grades[query.source_grade_id].model_dump(mode="json")
            if query.source_grade_id in self._grades
            else None,
            "source_metadata_claims": [
                claim.model_dump(mode="json") for claim in self._metadata_claims
            ],
            "records": [record.model_dump(mode="json") for record in records],
            "outcome_records": len(records),
            "formulations": len({r.formulation_id for r in records}),
            "polymer_present_formulations": len(
                {r.formulation_id for r in records if r.conditions.polymer_present is True}
            ),
            "zero_polymer_control_formulations": len(
                {r.formulation_id for r in records if r.conditions.polymer_present is False}
            ),
            "conflicting_literal_outcomes_at_same_reported_conditions": any(
                len(v) > 1 for v in outcomes.values()
            ),
            "filter_is_exact_reported_numeric_value_declared_unit_and_qualifier": True,
            "omitted_condition_filters_match_all_reported_values_and_unknowns": True,
            "explicit_null_filter_matches_only_unreported_value": True,
            "unknown_conditions_not_imputed": True,
            "nearest_match_interpolation_or_solubility_conversion": False,
        }
