"""Input representation is broader than the scientifically validated inference scope."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator
from rdkit import Chem

from .chemistry import molecule, normalize_smiles


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class MoleculeInput(StrictModel):
    kind: Literal["molecule"] = "molecule"
    smiles: str
    temperature_k: float = Field(default=298.15, gt=0)

    @field_validator("smiles")
    @classmethod
    def normalized(cls, value: str) -> str:
        return normalize_smiles(value)


class RepeatUnit(StrictModel):
    smiles: str
    mole_fraction: float = Field(gt=0, le=1)

    @field_validator("smiles")
    @classmethod
    def ports(cls, value: str) -> str:
        canonical = normalize_smiles(value)
        mol = molecule(canonical)
        ports = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(ports) != 2 or any(p.GetDegree() != 1 for p in ports):
            raise ValueError(
                "Repeat units require exactly two singly attached '*' connection ports"
            )
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Repeat unit must be connected")
        return canonical


class EndGroup(StrictModel):
    smiles: str
    count_per_chain: float = Field(default=1, gt=0, le=2)

    @field_validator("smiles")
    @classmethod
    def port(cls, value: str) -> str:
        canonical = normalize_smiles(value)
        mol = molecule(canonical)
        ports = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(ports) != 1 or ports[0].GetDegree() != 1:
            raise ValueError("End group requires one singly attached '*' connection port")
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("End group must be connected")
        return canonical


class EOPODistribution(StrictModel):
    eo_mean: float = Field(ge=0)
    po_mean: float = Field(ge=0)
    sequence: Literal["block", "random", "specified", "unknown"] = "unknown"
    distribution: Literal["monodisperse", "poisson", "empirical", "unknown"] = "unknown"
    eo_std: float | None = Field(default=None, ge=0)
    po_std: float | None = Field(default=None, ge=0)
    # Each entry is (EO count, PO count, mole probability), retaining correlation.
    joint_pmf: tuple[tuple[int, int, float], ...] = ()

    @model_validator(mode="after")
    def check_distribution(self):
        import math

        if self.eo_mean + self.po_mean <= 0:
            raise ValueError("EO/PO distribution must have nonzero chain length")
        if self.distribution == "empirical":
            if not self.joint_pmf:
                raise ValueError("Empirical distribution needs a joint PMF")
            if any(e < 0 or p < 0 or not math.isfinite(w) or w <= 0 for e, p, w in self.joint_pmf):
                raise ValueError("PMF counts must be nonnegative and probabilities positive/finite")
            if len({(e, p) for e, p, _ in self.joint_pmf}) != len(self.joint_pmf):
                raise ValueError("PMF bins must be unique")
            if abs(sum(w for _, _, w in self.joint_pmf) - 1) > 1e-6:
                raise ValueError("PMF probabilities must sum to one")
            for mean, pos in ((self.eo_mean, 0), (self.po_mean, 1)):
                if abs(sum(row[pos] * row[2] for row in self.joint_pmf) - mean) > 1e-6:
                    raise ValueError("Declared mean disagrees with joint PMF")
        elif self.joint_pmf:
            raise ValueError("joint_pmf is only allowed for an empirical distribution")
        if self.distribution == "monodisperse":
            if not self.eo_mean.is_integer() or not self.po_mean.is_integer():
                raise ValueError("Monodisperse repeat counts must be integers")
            if any(x not in (None, 0) for x in (self.eo_std, self.po_std)):
                raise ValueError("Monodisperse distribution has zero standard deviation")
        return self


class PolymerInput(StrictModel):
    kind: Literal["polymer"] = "polymer"
    series_id: str = Field(min_length=1)
    repeat_units: tuple[RepeatUnit, ...] = Field(min_length=1)
    mn_g_mol: float = Field(gt=0)
    dispersity: float | None = Field(default=None, ge=1)
    end_groups: tuple[EndGroup, ...] = Field(default=(), max_length=2)
    architecture: Literal["linear", "block", "random", "unknown"] = "unknown"
    eo_po: EOPODistribution | None = None
    temperature_k: float = Field(default=298.15, gt=0)

    @model_validator(mode="after")
    def fractions(self):
        if abs(sum(r.mole_fraction for r in self.repeat_units) - 1) > 1e-6:
            raise ValueError("Repeat-unit mole fractions must sum to one")
        if sum(g.count_per_chain for g in self.end_groups) > 2:
            raise ValueError("Linear-chain end-group count cannot exceed two")
        return self


class SurfactantInput(StrictModel):
    kind: Literal["surfactant"] = "surfactant"
    head_smiles: str
    tail_smiles: str
    eo_po: EOPODistribution
    mn_g_mol: float | None = Field(default=None, gt=0)
    ionic: bool = False
    counterion_smiles: str | None = None
    series_id: str = Field(min_length=1)
    temperature_k: float = Field(default=298.15, gt=0)

    @field_validator("head_smiles", "tail_smiles", "counterion_smiles")
    @classmethod
    def normalized(cls, value: str | None) -> str | None:
        return normalize_smiles(value) if value is not None else None

    @model_validator(mode="after")
    def counterion(self):
        if self.counterion_smiles is not None and not self.ionic:
            raise ValueError("A counterion requires ionic=True")
        return self


class MixtureInput(StrictModel):
    kind: Literal["mixture"] = "mixture"
    components: tuple[MoleculeInput, ...] = Field(min_length=2)
    mole_fractions: tuple[float, ...]

    @model_validator(mode="after")
    def fractions(self):
        import math

        if len(self.components) != len(self.mole_fractions):
            raise ValueError("One mole fraction is required per component")
        if any(not math.isfinite(x) or x <= 0 for x in self.mole_fractions):
            raise ValueError("Mole fractions must be positive and finite")
        if abs(sum(self.mole_fractions) - 1) > 1e-6:
            raise ValueError("Mole fractions must sum to one")
        return self


ChemicalInput = Annotated[
    MoleculeInput | PolymerInput | SurfactantInput | MixtureInput, Field(discriminator="kind")
]
INPUT_ADAPTER = TypeAdapter(ChemicalInput)
