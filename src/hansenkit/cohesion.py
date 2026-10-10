"""Read-only preparation of MD cohesive observables, not three-component HSP labels."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from .data import write_json
from .provenance import file_hash
from .schema import StrictModel

MD_UNITS = {
    "sp_ced": "J/cm^3",
    "sp_total": "MPa^0.5",
    "sp_vdw": "MPa^0.5",
    "sp_ele": "MPa^0.5",
}
REQUIRED_COLUMNS = {
    *MD_UNITS,
    "UUID",
    "temp",
    "press",
    "check_eq",
    "smiles_list",
    "forcefield",
    "RadonPy_ver",
    "preset_sp_ver",
}
MATERIAL_FIELDS = (
    "monomer_ID",
    "smiles_list",
    "smiles_1",
    "smiles_2",
    "smiles_3",
    "smiles_4",
    "smiles_ter_1",
    "smiles_ter_2",
    "ter_ID_1",
    "ter_ID_2",
    "copoly_ratio_list",
    "copoly_type",
    "DP",
    "Mn",
    "Mw",
    "Mw/Mn",
    "tacticity",
    "input_tacticity",
    "forcefield",
    "charge",
    "RadonPy_ver",
    "preset_eq_ver",
    "preset_sp_ver",
    "n_mol",
    "LAMMPS_ver",
    "RDKit_ver",
)


class MDSourceReview(StrictModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(min_length=1)
    declared_license: str = Field(min_length=1)
    permission_evidence: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    audit_basis: Literal["documented_permission", "operator_assumption"]
    rights_status: Literal["approved", "pending", "denied"]
    local_preparation_allowed: bool = False
    source_units: dict[str, str]
    definition_evidence: str = Field(min_length=1)
    training_allowed: Literal[False] = False
    public_redistribution: Literal[False] = False

    def authorize(self):
        if self.rights_status != "approved" or not self.local_preparation_allowed:
            raise ValueError("Explicit local MD preparation permission is required")
        if self.source_units != MD_UNITS:
            raise ValueError("Explicit source units must match the MD observable contract")


class MDCohesion(StrictModel):
    value_kind: Literal["computed_md_cohesion"] = "computed_md_cohesion"
    temperature_k: float = Field(gt=0)
    pressure_atm: float = Field(gt=0)
    sp_ced: float = Field(ge=0)
    sp_total: float = Field(ge=0)
    sp_vdw: float = Field(ge=0)
    sp_ele: float = Field(ge=0)

    def report(self):
        # Source values are separately averaged. Squaring a mean root is not mean energy.
        flags = []
        if self.sp_total * self.sp_total > self.sp_ced + 1e-6:
            flags.append("total_square_above_mean_ced")
        if self.sp_vdw * self.sp_vdw + self.sp_ele * self.sp_ele > self.sp_ced + 1e-6:
            flags.append("component_square_sum_above_mean_ced")
        return {
            **self.model_dump(),
            "units": MD_UNITS,
            "electrostatic_interpretation": "joint polar and hydrogen-bonding contribution",
            "individual_polar_hydrogen_components_identified": False,
            "hsp_predictions": None,
            "independent_experimental_accuracy_validated": False,
            "mean_energy_diagnostic_flags": flags,
        }


def prepare_md_cohesion(csv_path, review_path, out):
    """Preserve raw material metadata for later qualification; do not construct guessed chains."""
    review = MDSourceReview.model_validate_json(Path(review_path).read_text(encoding="utf-8"))
    review.authorize()
    if file_hash(csv_path) != review.sha256:
        raise ValueError("MD source checksum mismatch")
    directory = Path(out)
    if directory.exists():
        raise ValueError("MD output directory already exists; choose a fresh path")
    counts = Counter()
    flags = Counter()
    contexts = Counter()
    uuids = set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("CSV lacks MD observable/material metadata columns")
        directory.mkdir(parents=True)
        records_path = directory / "observations.jsonl"
        with records_path.open("x", encoding="utf-8") as records:
            for row in reader:
                counts["source_rows"] += 1
                if row["check_eq"] != "True":
                    counts["equilibrium_not_declared"] += 1
                    continue
                if any(
                    not row[key]
                    for key in ("UUID", "smiles_list", "forcefield", "RadonPy_ver", "preset_sp_ver")
                ):
                    counts["required_source_metadata_missing"] += 1
                    continue
                if row["UUID"] in uuids:
                    counts["repeated_uuid_refused"] += 1
                    continue
                try:
                    observation = MDCohesion(
                        temperature_k=row["temp"],
                        pressure_atm=row["press"],
                        **{key: row[key] for key in MD_UNITS},
                    ).report()
                except ValueError:
                    counts["missing_nonfinite_negative_observable_or_invalid_conditions"] += 1
                    continue
                uuids.add(row["UUID"])
                flags.update(observation["mean_energy_diagnostic_flags"])
                contexts[(row["temp"], row["press"], row["forcefield"], row["preset_sp_ver"])] += 1
                records.write(
                    json.dumps(
                        {
                            "source_uuid": row["UUID"],
                            "material_metadata": {
                                key: row.get(key) or None for key in MATERIAL_FIELDS
                            },
                            "material_identity_qualified": False,
                            "observation": observation,
                        }
                    )
                    + "\n"
                )
                counts["prepared_rows"] += 1
    report = {
        "label_kind": "computed_md_cohesion",
        "source_sha256": review.sha256,
        "review_sha256": file_hash(review_path),
        "source": review.source,
        "definition_evidence": review.definition_evidence,
        "units": MD_UNITS,
        "counts": dict(counts),
        "mean_energy_diagnostic_flags": dict(flags),
        "contexts": [
            {
                "temperature_k_source": t,
                "pressure_atm_source": p,
                "forcefield": ff,
                "preset_sp_ver": ver,
                "rows": n,
            }
            for (t, p, ff, ver), n in sorted(contexts.items())
        ],
        "prepared_sha256": file_hash(records_path),
        "audit_basis": review.audit_basis,
        "training_allowed": False,
        "public_redistribution": False,
        "material_identity_qualified": False,
        "holdout_family_isolation_qualified": False,
        "hsp_three_component_targets": 0,
        "physical_hsp_backend_qualified": False,
        "unknown_terminal_or_composition_fields_inferred": False,
    }
    write_json(directory / "preparation.json", report)
    return report
