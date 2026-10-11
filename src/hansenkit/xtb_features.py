"""Strict opt-in parsing of reviewed gas GFN2 outputs; never HSP prediction.

No execution, downloads, optimization, value clipping, or force-field fallback is
performed here. Runtime-file verification belongs to the reviewed caller before
execution. Constants follow the binary's resolved 5071a88 conversion source.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors

from .charge import charge_classification
from .chemistry import normalize_smiles

VERSION = "6.7.1pre (5071a88)"
_MAIN_GROUP_ELEMENTS = frozenset((1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53))
# https://github.com/grimme-lab/xtb/blob/5071a8869720fb0ecd788af977159869b64a277c/src/mctc/convert.f90
BOHR_ANGSTROM = 0.52917726
DIPOLE_AU_TO_DEBYE = 1.6021766208e-19 * 137.0359990740 * BOHR_ANGSTROM**2 * 41.3413733365614 * 1e16
GAS_PROTOCOL = {
    "method": "GFN2-xTB",
    "phase": "gas",
    "charge": 0,
    "unpaired_electrons": 0,
    "accuracy": 1.0,
    "maximum_scc_iterations": 250,
    "electronic_temperature_k": 300.0,
    "restart": False,
    "geometry_optimization": False,
    "threads_per_process": 1,
    "maximum_concurrent_processes": 4,
    "timeout_seconds_per_conformer": 120,
    "conformers": 3,
    "all_conformers_required": True,
    "maximum_heavy_atoms": 60,
    "maximum_molecular_weight": 500.0,
    "fallback": None,
    "expected_xtb_version": VERSION,
    "qualified_hsp_backend": False,
}
SCALAR_NAMES = (
    "gfn2_scc_full_dipole_debye",
    "gfn2_gap_ev",
    "gfn2_d4_alpha_bohr3",
    "gfn2_d4_c6_hartree_bohr6",
    "gfn2_d4_c8_hartree_bohr8",
)
SCALAR_UNITS = ("Debye", "eV", "a0^3", "Hartree a0^6", "Hartree a0^8")
FEATURE_NAMES = tuple(f"{name}_{stat}" for name in SCALAR_NAMES for stat in ("mean", "std")) + (
    "gfn2_d4_alpha_per_geometric_volume_mean",
    "gfn2_d4_c6_per_geometric_volume_squared_mean",
    "gfn2_scc_full_dipole_squared_per_geometric_volume_mean",
    "gfn2_abs_charge_per_heavy_mean",
    "gfn2_charge_squared_per_heavy_mean",
    "gfn2_max_charge_mean",
    "gfn2_min_charge_mean",
    "gfn2_donor_h_charge_sum_per_heavy_mean",
    "gfn2_acceptor_charge_sum_per_heavy_mean",
    "gfn2_hetero_abs_charge_per_heavy_mean",
)
FEATURE_UNITS = tuple(unit for unit in SCALAR_UNITS for _ in range(2)) + (
    "a0^3 / Angstrom^3",
    "Hartree a0^6 / Angstrom^6",
    "Debye^2 / Angstrom^3",
    "e / heavy_atom",
    "e^2 / heavy_atom",
    "e",
    "e",
    "e / heavy_atom",
    "e / heavy_atom",
    "e / heavy_atom",
)
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a scientific number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Scientific scalar number required") from error
    if not np.isfinite(converted):
        raise ValueError("Nonfinite scientific number")
    return converted


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _decode(raw, repair):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value}")

    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 10 * 1024 * 1024:
        raise ValueError("Bounded text JSON required")
    diagnostic = {"json_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
    try:
        data = json.loads(
            raw, object_pairs_hook=_unique, parse_constant=invalid, parse_float=_number
        )
        diagnostic["metadata_program_call_removed"] = False
    except json.JSONDecodeError as error:
        lines = raw.splitlines(keepends=True)
        matches = [
            i for i, line in enumerate(lines) if re.fullmatch(r'\s*"program call": ".*",\s*', line)
        ]
        if not repair or len(matches) != 1 or error.lineno != matches[0] + 1:
            raise ValueError("Malformed xTB JSON; no scientific value repair") from error
        index = matches[0]
        data = json.loads(
            "".join(lines[:index] + lines[index + 1 :]),
            object_pairs_hook=_unique,
            parse_constant=invalid,
            parse_float=_number,
        )
        diagnostic.update(
            metadata_program_call_removed=True,
            metadata_removal_reason="single malformed Windows program-call metadata line",
            removed_metadata_line_sha256=hashlib.sha256(lines[index].encode("utf-8")).hexdigest(),
        )
    if not isinstance(data, dict):
        raise ValueError("xTB JSON object required")
    return data, diagnostic


def _one(text, pattern):
    values = re.findall(pattern, text, re.MULTILINE)
    if len(values) != 1:
        raise ValueError("Require one unambiguous singlepoint output field")
    return values[0]


def _printed_number(text, label):
    result = _one(text, rf"^\s*{re.escape(label)}\s*:\s*({_FLOAT})\s*$")
    return _number(result.replace("D", "E").replace("d", "e"))


def _review(review, digest):
    if (
        not isinstance(review, dict)
        or review.get("audit_basis") != "operator_assumption"
        or review.get("scope") != "local_scientific_private_noncommercial"
        or review.get("execution_allowed") is not True
        or review.get("redistribution_allowed") is not False
        or review.get("qualified_hsp_backend") is not False
        or review.get("protocol") != GAS_PROTOCOL
        or not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest) is None
    ):
        raise ValueError("Exact gas protocol and scoped raw-file review SHA-256 required")


@lru_cache(maxsize=1)
def _factory():
    return ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))


@dataclass(frozen=True)
class XTBConformer:
    canonical_smiles: str
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    scalars: np.ndarray
    charges: np.ndarray
    full_dipole_au: np.ndarray
    atomic_dipoles_au: np.ndarray
    donor_hydrogen_indices: tuple
    acceptor_indices: tuple
    diagnostics: dict

    @property
    def mu_debye(self):
        return float(self.scalars[0])

    @property
    def gap_ev(self):
        return float(self.scalars[1])

    @property
    def alpha_bohr3(self):
        return float(self.scalars[2])

    @property
    def c6_hartree_bohr6(self):
        return float(self.scalars[3])

    @property
    def c8_hartree_bohr8(self):
        return float(self.scalars[4])

    def to_dict(self):
        return {
            "artifact_type": "computed_gas_gfn2_conformer_features",
            "canonical_smiles": self.canonical_smiles,
            "atomic_numbers": self.atomic_numbers.tolist(),
            "positions_angstrom": self.positions_angstrom.tolist(),
            "scalar_names": list(SCALAR_NAMES),
            "scalar_units": list(SCALAR_UNITS),
            "scalars": self.scalars.tolist(),
            "charges_e": self.charges.tolist(),
            "full_dipole_au": self.full_dipole_au.tolist(),
            "atomic_dipoles_au": self.atomic_dipoles_au.tolist(),
            "donor_hydrogen_indices": list(self.donor_hydrogen_indices),
            "acceptor_indices": list(self.acceptor_indices),
            "diagnostics": self.diagnostics,
        }


def parse_xtb_singlepoint(
    stdout,
    stderr,
    json_text,
    *,
    smiles,
    atomic_numbers,
    positions_angstrom,
    review,
    review_sha256,
    returncode=0,
    repair_program_call=False,
):
    """Validate one complete neutral gas singlepoint. Never executes a process.

    ``review_sha256`` identifies the caller's raw review-file bytes; this function
    does not serialize/re-hash the dict or certify assets checked by the caller.
    Optional malformed-metadata handling removes only one program-call line and
    preserves its original hash. All scientific JSON remains strictly parsed.
    """
    _review(review, review_sha256)
    if returncode != 0 or not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ValueError("Successful reviewed process and complete text traces required")
    if (
        len(re.findall(r"normal termination of xtb", stderr)) != 1
        or len(re.findall(r"convergence criteria satisfied after \d+ iterations", stdout)) != 1
        or re.search(
            r"\bERROR\b|failed to converge|abnormal termination|SCC not converged",
            stdout + stderr,
            re.I,
        )
        or "GEOMETRY OPTIMIZATION" in stdout.upper()
    ):
        raise ValueError("One converged singlepoint with normal termination required")
    setup = (
        (r"Hamiltonian\s+GFN2-xTB\s+:",),
        (r"restarted\?\s+false\s+:",),
        (r"GBSA solvation\s+false\s+:",),
        (r"PC potential\s+false\s+:",),
        (r"max\. iterations\s+250\s+:",),
        (r"electronic temp\.\s+300\.0+\s+K\s+:",),
        (r"accuracy\s+1\.0+\s+:",),
        (r"net charge\s+0\s+:",),
        (r"unpaired electrons\s+0\s+:",),
    )
    if any(len(re.findall(pattern[0], stdout)) != 1 for pattern in setup):
        raise ValueError("Output differs from the exact gas protocol")
    canonical = normalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if (
        charge_classification(mol) == "unreviewed_charge_or_fragments"
        or not 1 <= mol.GetNumHeavyAtoms() <= 60
        or Descriptors.MolWt(mol) > 500
        or any(
            a.GetNumRadicalElectrons() or a.GetIsotope() or a.GetAtomicNum() == 0
            for a in mol.GetAtoms()
        )
    ):
        raise ValueError("Require an applicable neutral connected bounded graph")
    explicit = Chem.AddHs(Chem.Mol(mol))
    expected = np.array([a.GetAtomicNum() for a in explicit.GetAtoms()])
    if not all(int(z) in _MAIN_GROUP_ELEMENTS for z in expected):
        raise ValueError("Valence electron accounting requires reviewed main-group elements")
    numbers = np.asarray(atomic_numbers)
    positions = np.asarray(positions_angstrom, float)
    if (
        numbers.dtype.kind not in "iu"
        or not np.array_equal(numbers, expected)
        or positions.shape != (len(expected), 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Require complete original atom order and finite Angstrom coordinates")
    payload, diagnostic = _decode(json_text, repair_program_call)
    if payload.get("method") != "GFN2-xTB" or payload.get("xtb version") != VERSION:
        raise ValueError("Unreviewed xTB method or binary version")
    valence_electrons = sum(Chem.GetPeriodicTable().GetNOuterElecs(int(z)) for z in numbers)
    for name, value in (
        ("number of electrons", valence_electrons),
        ("number of unpaired electrons", 0),
    ):
        if isinstance(payload.get(name), bool) or payload.get(name) != value:
            raise ValueError("Electron count does not match the complete neutral graph")
    charges = np.asarray(payload.get("partial charges"), float)
    dipole = np.asarray(payload.get("dipole / a.u."), float)
    atomic_dipoles = np.asarray(payload.get("atomic dipole moments"), float)
    if (
        charges.shape != numbers.shape
        or dipole.shape != (3,)
        or atomic_dipoles.shape != (len(numbers), 3)
        or not all(np.isfinite(x).all() for x in (charges, dipole, atomic_dipoles))
        or abs(charges.sum()) > 1e-5
    ):
        raise ValueError("Complete finite neutral charges and full multipoles required")
    # SCC-result JSON and final wavefunction multipoles are separate source states.
    # Validate final-wavefunction reconstruction against its three-decimal printout.
    q_only = charges @ (positions / BOHR_ANGSTROM)
    reconstructed = q_only + atomic_dipoles.sum(0)
    printed_q = np.array(
        [
            _number(x)
            for x in _one(stdout, rf"^\s*q only:\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$")
        ]
    )
    printed_full = np.array(
        [
            _number(x)
            for x in _one(
                stdout, rf"^\s*full:\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$"
            )
        ]
    )
    if (
        np.any(np.abs(q_only - printed_q) > 0.000501)
        or np.any(np.abs(reconstructed - printed_full[:3]) > 0.000501)
        or abs(np.linalg.norm(reconstructed) * DIPOLE_AU_TO_DEBYE - printed_full[3]) > 0.000501
    ):
        raise ValueError(
            "Final wavefunction full/q-only dipoles disagree with atom positions and multipoles"
        )
    table = _one(stdout, r"^\s*#\s+Z\s+covCN\s+q\s+C6AA\s+α\(0\)\s*\n((?:\s*\d+[^\n]*\n)+)")
    population = []
    for line in table.splitlines():
        match = re.fullmatch(
            rf"\s*(\d+)\s+(\d+)\s+[A-Z][a-z]?\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*",
            line,
        )
        if match is None:
            raise ValueError("Incomplete atom population table")
        population.append([_number(x.replace("D", "E").replace("d", "e")) for x in match.groups()])
    population = np.asarray(population)
    if (
        population.shape != (len(numbers), 6)
        or not np.array_equal(population[:, 0], np.arange(1, len(numbers) + 1))
        or not np.array_equal(population[:, 1], numbers)
        or np.any(np.abs(population[:, 3] - charges) > 0.000501)
        or np.any(population[:, 4:] < 0)
    ):
        raise ValueError("Population table differs from complete atomic identities or charges")
    scalars = np.array(
        [
            np.linalg.norm(dipole) * DIPOLE_AU_TO_DEBYE,
            _number(payload.get("HOMO-LUMO gap / eV")),
            _printed_number(stdout, "Mol. α(0) /au"),
            _printed_number(stdout, "Mol. C6AA /au·bohr⁶"),
            _printed_number(stdout, "Mol. C8AA /au·bohr⁸"),
        ]
    )
    if np.any(scalars < 0) or np.any(scalars[2:] <= 0) or not np.isfinite(scalars).all():
        raise ValueError("Invalid nonnegative gas observables; no clipping is permitted")
    features = _factory().GetFeaturesForMol(mol)
    donors = {i for f in features if f.GetFamily() == "Donor" for i in f.GetAtomIds()}
    acceptors = tuple(
        sorted({i for f in features if f.GetFamily() == "Acceptor" for i in f.GetAtomIds()})
    )
    donor_h = tuple(
        sorted(
            {
                h.GetIdx()
                for d in donors
                for h in explicit.GetAtomWithIdx(d).GetNeighbors()
                if h.GetAtomicNum() == 1
            }
        )
    )
    diagnostic.update(
        review_sha256=review_sha256,
        audit_basis="operator_assumption",
        scope="local_scientific_private_noncommercial",
        redistribution_allowed=False,
        qualified_hsp_backend=False,
        hsp_predictions=None,
        hydrogen_bond_strength_measured=False,
        observable_kind="computed_gas_gfn2_structure_features",
        method="GFN2-xTB",
        version=VERSION,
        protocol=GAS_PROTOCOL.copy(),
        source_smiles=canonical,
        stdout_sha256=hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        scalar_units=list(SCALAR_UNITS),
        charge_unit="e",
        dipole_unit="e a0",
        dipole_model="scc_result_full_electronic_multipoles",
        dispersion_model="GFN2_self_consistent_D4",
        scc_vs_final_wavefunction_dipole_max_abs_au=float(np.abs(reconstructed - dipole).max()),
        final_wavefunction_full_dipole_au=reconstructed.tolist(),
        multipole_closure="final_wavefunction_verified_against_three_decimal_stdout",
        q_only_dipole_au=q_only.tolist(),
        topology_disagreement_reported="Topologies differ" in stdout,
    )
    return XTBConformer(
        canonical,
        numbers.copy(),
        positions.copy(),
        scalars,
        charges,
        dipole,
        atomic_dipoles,
        donor_h,
        acceptors,
        diagnostic,
    )


@dataclass(frozen=True)
class XTBFeatures:
    values: np.ndarray
    diagnostics: dict

    @property
    def names(self):
        return FEATURE_NAMES

    @property
    def units(self):
        return FEATURE_UNITS

    def to_dict(self):
        return {
            "names": list(self.names),
            "units": list(self.units),
            "values": self.values.tolist(),
            "diagnostics": self.diagnostics,
        }


def aggregate_xtb_conformers(records, *, volumes_angstrom3):
    """All three complete original conformers, descriptive statistics only."""
    if len(records) != 3 or not all(isinstance(r, XTBConformer) for r in records):
        raise ValueError("Exactly three complete parsed conformers required")
    first = records[0]
    for record in records:
        if (
            record.canonical_smiles != first.canonical_smiles
            or not np.array_equal(record.atomic_numbers, first.atomic_numbers)
            or record.donor_hydrogen_indices != first.donor_hydrogen_indices
            or record.acceptor_indices != first.acceptor_indices
            or record.diagnostics.get("protocol") != GAS_PROTOCOL
            or record.diagnostics.get("review_sha256") != first.diagnostics.get("review_sha256")
            or record.diagnostics.get("qualified_hsp_backend") is not False
            or record.diagnostics.get("redistribution_allowed") is not False
            or record.diagnostics.get("audit_basis") != "operator_assumption"
            or record.diagnostics.get("version") != VERSION
            or record.diagnostics.get("method") != "GFN2-xTB"
            or record.diagnostics.get("scalar_units") != list(SCALAR_UNITS)
            or record.diagnostics.get("scope") != "local_scientific_private_noncommercial"
            or record.scalars.shape != (5,)
            or not np.isfinite(record.scalars).all()
            or np.any(record.scalars < 0)
            or np.any(record.scalars[2:] <= 0)
            or record.charges.shape != first.atomic_numbers.shape
            or not np.isfinite(record.charges).all()
            or abs(record.charges.sum()) > 1e-5
            or record.full_dipole_au.shape != (3,)
            or not np.isfinite(record.full_dipole_au).all()
            or not np.isclose(
                record.mu_debye,
                np.linalg.norm(record.full_dipole_au) * DIPOLE_AU_TO_DEBYE,
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError("Conformer graph, review, protocol or complete values differ")
    volume = np.asarray(volumes_angstrom3, float)
    if volume.shape != (3,) or not np.isfinite(volume).all() or np.any(volume <= 0):
        raise ValueError("Complete positive geometric Angstrom^3 volumes required")
    scalar = np.array([r.scalars for r in records])
    summaries = np.column_stack([scalar.mean(0), scalar.std(0)]).ravel()
    ratios = np.array(
        [
            (scalar[:, 2] / volume).mean(),
            (scalar[:, 3] / volume**2).mean(),
            (scalar[:, 0] ** 2 / volume).mean(),
        ]
    )
    n = int(np.count_nonzero(first.atomic_numbers != 1))
    hetero = ~np.isin(first.atomic_numbers, (1, 6))
    charge_rows = []
    for record in records:
        q = record.charges
        charge_rows.append(
            [
                np.abs(q).sum() / n,
                (q**2).sum() / n,
                q.max(),
                q.min(),
                q[list(record.donor_hydrogen_indices)].sum() / n,
                q[list(record.acceptor_indices)].sum() / n,
                np.abs(q[hetero]).sum() / n,
            ]
        )
    values = np.r_[summaries, ratios, np.mean(charge_rows, axis=0)]
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Invalid complete conformer aggregation")
    diagnostic = {
        "artifact_type": "computed_gas_gfn2_structure_features",
        "canonical_smiles": first.canonical_smiles,
        "review_sha256": first.diagnostics["review_sha256"],
        "audit_basis": "operator_assumption",
        "scope": "local_scientific_private_noncommercial",
        "redistribution_allowed": False,
        "qualified_hsp_backend": False,
        "hsp_predictions": None,
        "hydrogen_bond_strength_measured": False,
        "equilibrium_qualified": False,
        "conformer_statistics": "equal_weight_descriptive",
        "protocol": GAS_PROTOCOL.copy(),
        "conformer_diagnostics": [r.diagnostics for r in records],
        "geometric_volume_unit": "Angstrom^3",
        "molar_liquid_volume": False,
        "units": list(FEATURE_UNITS),
    }
    return XTBFeatures(values, diagnostic)
