"""Opt-in coefficient-reference arithmetic; no validated physical HSP backend."""

import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from rdkit import Chem

from .chains import assemble_homopolymer
from .chemistry import normalize_smiles
from .provenance import file_hash
from .schema import StrictModel

GROUPS = frozenset({"methyl", "methylene", "methine", "vinyl_methine", "ester", "alcohol"})
GROUP_VERSION = "original-six-acyclic-CO-types-v1"
MAX_EXPLICIT_HEAVY_ATOMS = 512
COEFFICIENT_UNITS = {
    "fd": "sqrt(J)*cm^(3/2)/mol",
    "fp": "sqrt(J)*cm^(3/2)/mol",
    "eh": "J/mol",
    "volume": "cm^3/mol",
}


class GroupCoefficients(StrictModel):
    fd: float = Field(ge=0)
    fp: float = Field(ge=0)
    eh: float = Field(ge=0)
    # Negative partial volumes are permitted; the total must remain positive.
    volume: float


class CoefficientDataset(StrictModel):
    units: dict[str, str]
    polar_convention: Literal["rss_of_group_type_force_sums"]
    groups: dict[str, GroupCoefficients] = Field(min_length=1)
    reference_temperature_k: float | None

    @model_validator(mode="after")
    def check_contract(self):
        if self.units != COEFFICIENT_UNITS or set(self.groups) - GROUPS:
            raise ValueError("Unsupported coefficient units or group types")
        if self.reference_temperature_k is not None and self.reference_temperature_k <= 0:
            raise ValueError(
                "Coefficient reference temperature must be positive or explicitly unknown"
            )
        return self


class CoefficientReview(StrictModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(min_length=1)
    declared_license: str = Field(min_length=1)
    permission_evidence: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    audit_basis: Literal["documented_permission", "operator_assumption"]
    rights_status: Literal["approved", "pending", "denied"]
    local_reference_calculation_allowed: bool = False
    training_or_pseudo_label_generation_allowed: Literal[False] = False
    public_redistribution: Literal[False] = False

    def authorize(self):
        if self.rights_status != "approved" or not self.local_reference_calculation_allowed:
            raise ValueError("Explicit local coefficient-reference permission is required")


def source_group_counts(smiles: str) -> dict[str, int]:
    """Type every heavy atom without inserting a missing physical group."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    if len(smiles) > 8192:
        raise ValueError("Use bounded repeat-count calculation for very large structures")
    # Check before canonical serialization, which can exhaust native Windows stacks.
    with Chem.rdBase.BlockLogs():
        parsed = Chem.MolFromSmiles(smiles)
    if parsed is None or parsed.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES")
    if parsed.GetNumHeavyAtoms() > MAX_EXPLICIT_HEAVY_ATOMS:
        raise ValueError("Explicit structure exceeds heavy-atom budget; use bounded repeat count")
    mol = Chem.MolFromSmiles(normalize_smiles(smiles))
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings():
        raise ValueError("Source groups require one acyclic molecule")
    if any(
        a.GetIsAromatic()
        or a.GetFormalCharge()
        or a.GetNumRadicalElectrons()
        or a.GetIsotope()
        or not a.GetAtomicNum()
        for a in mol.GetAtoms()
    ):
        raise ValueError(
            "Source groups do not support aromatic, charged, radical, isotope or port atoms"
        )
    owners = {}
    counts = Counter()
    for name, smarts, owned in (
        ("ester", "[CX3](=[OX1])[OX2H0][#6]", (0, 1, 2)),
        ("alcohol", "[OX2H1][CX4]", (0,)),
    ):
        for match in sorted(
            mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=True, maxMatches=0)
        ):
            atoms = tuple(match[i] for i in owned)
            if any(i in owners for i in atoms):
                continue
            owners.update(dict.fromkeys(atoms, name))
            counts[name] += 1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 or atom.GetIdx() in owners:
            continue
        name = None
        if atom.GetAtomicNum() == 6:
            hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                name = {3: "methyl", 2: "methylene", 1: "methine"}.get(hydrogens)
            elif (
                atom.GetHybridization() == Chem.HybridizationType.SP2
                and hydrogens == 1
                and sum(
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetOtherAtom(atom).GetAtomicNum() == 6
                    for bond in atom.GetBonds()
                )
                == 1
            ):
                name = "vinyl_methine"
        if name is None:
            raise ValueError("Unassigned heavy atom: source coefficient coverage is incomplete")
        owners[atom.GetIdx()] = name
        counts[name] += 1
    if len(owners) != mol.GetNumHeavyAtoms():
        raise ValueError("Incomplete atom ownership")
    return dict(counts)


class ReferenceGC:
    """A source-convention reference calculator, separate from HSPModel inference."""

    def __init__(self, parameters, review):
        self.review = CoefficientReview.model_validate_json(
            Path(review).read_text(encoding="utf-8")
        )
        self.review.authorize()
        if Path(parameters).stat().st_size > 65_536 or file_hash(parameters) != self.review.sha256:
            raise ValueError("Coefficient file size/checksum mismatch")
        self.parameters = CoefficientDataset.model_validate_json(
            Path(parameters).read_text(encoding="utf-8")
        )

    def _calculate(self, counts, basis):
        if set(counts) - self.parameters.groups.keys():
            raise ValueError("Required source-group coefficient is missing")
        if not counts or any(not math.isfinite(n) or n < 0 for n in counts.values()):
            raise ValueError("Invalid group population")
        p = self.parameters.groups
        volume = math.fsum(n * p[g].volume for g, n in counts.items())
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Total source-group volume is nonpositive or nonfinite")
        fd = math.fsum(n * p[g].fd for g, n in counts.items())
        fp = math.hypot(*(n * p[g].fp for g, n in counts.items()))
        eh = math.fsum(n * p[g].eh for g, n in counts.items())
        values = [fd / volume, fp / volume, math.sqrt(eh / volume)]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("Nonfinite source-reference calculation")
        return {
            "value_kind": "coefficient_reference_hsp",
            "delta_d": values[0],
            "delta_p": values[1],
            "delta_h": values[2],
            "units": "MPa^0.5",
            "group_count_basis": basis,
            "group_coverage": 1.0,
            "group_version": GROUP_VERSION,
            "coefficient_sha256": self.review.sha256,
            "source": self.review.source,
            "polar_convention": self.parameters.polar_convention,
            "reference_temperature_k": self.parameters.reference_temperature_k,
            "physical_accuracy_validated": False,
            "calibrated_uncertainty": None,
            "training_or_pseudo_label_generation_allowed": False,
        }

    def molecule(self, smiles):
        return self._calculate(source_group_counts(smiles), "one molecule")

    def homopolymer(self, repeat_unit, n, *, left_cap, right_cap):
        """Caps must be declared; explicit None means hydrogen, as in the assembler."""
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10**12:
            raise ValueError("Declared repeat count must be an integer within 1–1e12")
        if n <= 64:
            result = self.molecule(assemble_homopolymer(repeat_unit, n, left_cap, right_cap))
            result["repeat_count"] = n
            return result
        samples = [
            source_group_counts(assemble_homopolymer(repeat_unit, count, left_cap, right_cap))
            for count in (4, 16, 64)
        ]
        keys = set().union(*samples)
        normalized = {}
        for key in keys:
            a, b, c = [sample.get(key, 0) for sample in samples]
            slope = (b - a) / 12
            if slope < 0 or not math.isclose(slope, (c - b) / 48, abs_tol=1e-12):
                raise ValueError("Group additivity is not certified for this repeat/cap structure")
            normalized[key] = slope + (a - 4 * slope) / n
        result = self._calculate(normalized, "per repeat, including declared cap contribution")
        result["bounded_additivity_checks"] = [4, 16, 64]
        result["repeat_count"] = n
        return result
