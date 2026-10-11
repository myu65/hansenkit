"""Opt-in frame-matched nonbonded cohesion. Does not predict Hansen components."""

from dataclasses import dataclass, field
from math import pi

import numpy as np

OBSERVER_VERSION = "openmm-frame-nonbonded-subtraction-v3"
# SI constants matching the pinned OpenMM 8.6.1 reference implementation.
AVOGADRO = 6.02214076e23
COULOMB_K = 1 / (4 * pi * (1e-6 * 8.8541878128e-12 / (1.602176634e-19**2 * AVOGADRO)))


def energy_density_mpa(energy_kj_mol: float, volume_nm3: float) -> float:
    """Energy of one simulation cell / its volume; retain signed density."""
    if not np.isfinite(energy_kj_mol) or not np.isfinite(volume_nm3) or volume_nm3 <= 0:
        raise ValueError("Finite energy and positive finite cell volume are required")
    return float(energy_kj_mol / volume_nm3 * 1e24 / AVOGADRO)


@dataclass(frozen=True)
class CohesionFrame:
    volume_nm3: float
    bulk_lj_kj_mol: float
    bulk_coulomb_kj_mol: float
    isolated_lj_kj_mol: float
    isolated_coulomb_kj_mol: float
    ced_lj_mpa: float
    ced_coulomb_mpa: float
    ced_nonbonded_mpa: float
    bulk_split_closure_error_kj_mol: float
    bulk_split_closure_tolerance_kj_mol: float
    observable_kind: str = field(default="computed_frame_nonbonded_cohesion", init=False)
    physical_accuracy_validated: bool = field(default=False, init=False)
    hsp_component_split_identified: bool = field(default=False, init=False)
    hsp_predictions: None = field(default=None, init=False)


class FrameCohesionObserver:
    """Evaluate an already parameterized neutral molecular cell on CPU/Reference.

    The caller reviews its parameter assets and records material/ensemble provenance.
    The observer never integrates dynamics, downloads assets or reads model files.
    PME bulk energies include the original periodic/tail corrections. Isolated
    molecules retain their instantaneous conformations and original 1-4 exceptions.
    """

    def __init__(self, system, molecule_ids, bonds, *, platform="CPU", threads=2):
        import openmm as mm
        from openmm import unit

        if platform not in {"CPU", "Reference"}:
            raise ValueError("This reviewed observer supports explicit CPU or Reference only")
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError("CPU thread count must be an integer between 1 and 8")
        n = system.getNumParticles()
        ids = np.asarray(molecule_ids)
        if not 2 <= n <= 8192 or ids.shape != (n,) or ids.dtype.kind not in "iu":
            raise ValueError("Provide 2..8192 particles and one integer molecule ID per particle")
        unique = np.unique(ids)
        if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError("At least two molecules with contiguous IDs starting at zero required")
        if any(system.isVirtualSite(i) for i in range(n)):
            raise ValueError("Virtual-site and polarizable models need another observer")
        forces = system.getForces()
        allowed = {
            "NonbondedForce",
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "CMMotionRemover",
            "MonteCarloBarostat",
        }
        if any(type(force).__name__ not in allowed for force in forces):
            raise ValueError("Force model outside the explicitly supported additive profile")
        nonbonded = [force for force in forces if isinstance(force, mm.NonbondedForce)]
        if len(nonbonded) != 1 or nonbonded[0].getNumParticles() != n:
            raise ValueError("Exactly one complete NonbondedForce is required")
        source = nonbonded[0]
        if source.getNonbondedMethod() not in {mm.NonbondedForce.NoCutoff, mm.NonbondedForce.PME}:
            raise ValueError("Only PME cells or explicit NoCutoff analytic fixtures are supported")
        if (
            source.getNumParticleParameterOffsets()
            or source.getNumExceptionParameterOffsets()
            or not source.getIncludeDirectSpace()
            or source.getReciprocalSpaceForceGroup() != -1
        ):
            raise ValueError("Parameter-offset and split/reciprocal-only models are not supported")
        particles = [source.getParticleParameters(i) for i in range(n)]
        numeric = np.array(
            [
                [
                    q.value_in_unit(unit.elementary_charge),
                    sigma.value_in_unit(unit.nanometer),
                    epsilon.value_in_unit(unit.kilojoule_per_mole),
                ]
                for q, sigma, epsilon in particles
            ]
        )
        if not np.isfinite(numeric).all() or (numeric[:, 1:] < 0).any():
            raise ValueError("Nonfinite or negative nonbonded parameters")
        self._groups = [np.where(ids == group)[0].tolist() for group in unique]
        if any(abs(numeric[group, 0].sum()) > 1e-4 for group in self._groups):
            raise ValueError("Each molecule must have neutral partial-charge sum")
        self._neighbors = [[] for _ in range(n)]
        edges = set()
        for a, b in bonds:
            if (
                not isinstance(a, (int, np.integer))
                or not isinstance(b, (int, np.integer))
                or isinstance(a, (bool, np.bool_))
                or isinstance(b, (bool, np.bool_))
                or not 0 <= a < n
                or not 0 <= b < n
                or a == b
                or ids[a] != ids[b]
            ):
                raise ValueError("Bond indices must connect distinct atoms within one molecule")
            pair = tuple(sorted((int(a), int(b))))
            if pair not in edges:
                edges.add(pair)
                self._neighbors[a].append(b)
                self._neighbors[b].append(a)
        for force in forces:
            if isinstance(force, mm.HarmonicBondForce):
                for index in range(force.getNumBonds()):
                    a, b, *_ = force.getBondParameters(index)
                    if tuple(sorted((a, b))) not in edges:
                        raise ValueError("Supplied topology omits a force-field bond")
        for index in range(system.getNumConstraints()):
            a, b, _ = system.getConstraintParameters(index)
            if ids[a] != ids[b]:
                raise ValueError("A constraint crosses declared molecule boundaries")
        for group in self._groups:
            reached, pending = {group[0]}, [group[0]]
            while pending:
                for neighbor in self._neighbors[pending.pop()]:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)
            if reached != set(group):
                raise ValueError("Every molecule must be one connected supplied bond component")
        exceptions = [source.getExceptionParameters(i) for i in range(source.getNumExceptions())]
        if any(ids[a] != ids[b] for a, b, *_ in exceptions):
            raise ValueError("An exception crosses declared molecule boundaries")
        self._periodic = source.getNonbondedMethod() == mm.NonbondedForce.PME
        self._cutoff_nm = source.getCutoffDistance().value_in_unit(unit.nanometer)
        self._n, self._unit = n, unit
        bulk, gas = mm.System(), mm.System()
        bulk.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        for i in range(n):
            mass = system.getParticleMass(i)
            if not np.isfinite(mass.value_in_unit(unit.dalton)) or mass <= 0 * unit.dalton:
                raise ValueError("Each real particle must have positive finite mass")
            bulk.addParticle(mass)
            gas.addParticle(mass)
        for group in range(3):
            force = mm.XmlSerializer.clone(source)
            force.setForceGroup(group)
            if group < 2:
                for i, (q, sigma, epsilon) in enumerate(particles):
                    force.setParticleParameters(i, q * (group == 1), sigma, epsilon * (group == 0))
                for i, (a, b, qprod, sigma, epsilon) in enumerate(exceptions):
                    force.setExceptionParameters(
                        i, a, b, qprod * (group == 1), sigma, epsilon * (group == 0)
                    )
            bulk.addForce(force)
        for group, expression in enumerate(
            [
                "4*sqrt(epsilon1*epsilon2)*((sigma/r)^12-(sigma/r)^6);sigma=0.5*(sigma1+sigma2)",
                f"{COULOMB_K:.17g}*q1*q2/r",
            ]
        ):
            force = mm.CustomNonbondedForce(expression)
            force.setForceGroup(group)
            for name in ["q", "sigma", "epsilon"]:
                force.addPerParticleParameter(name)
            for parameters in numeric:
                force.addParticle(parameters.tolist())
            for members in self._groups:
                force.addInteractionGroup(set(members), set(members))
            for a, b, *_ in exceptions:
                force.addExclusion(a, b)
            gas.addForce(force)
            exception_force = mm.CustomBondForce(
                "4*epsilon*((sigma/r)^12-(sigma/r)^6)"
                if group == 0
                else f"{COULOMB_K:.17g}*qprod/r"
            )
            exception_force.setForceGroup(group)
            for name in ["qprod", "sigma", "epsilon"]:
                exception_force.addPerBondParameter(name)
            for a, b, qprod, sigma, epsilon in exceptions:
                exception_force.addBond(
                    a,
                    b,
                    [
                        qprod.value_in_unit(unit.elementary_charge**2),
                        sigma.value_in_unit(unit.nanometer),
                        epsilon.value_in_unit(unit.kilojoule_per_mole),
                    ],
                )
            gas.addForce(exception_force)
        chosen = mm.Platform.getPlatformByName(platform)
        properties = (
            {"Threads": str(threads), "DeterministicForces": "true"} if platform == "CPU" else {}
        )
        self._bulk_integrator, self._gas_integrator = (
            mm.VerletIntegrator(0.001),
            mm.VerletIntegrator(0.001),
        )
        self._bulk_context = mm.Context(bulk, self._bulk_integrator, chosen, properties)
        self._gas_context = mm.Context(gas, self._gas_integrator, chosen, properties)
        self._closed = False
        self.provenance = {
            "observer_version": OBSERVER_VERSION,
            "openmm_version": mm.__version__,
            "platform": platform,
            "molecules": len(self._groups),
            "particles": n,
            "periodic": self._periodic,
            "gas_definition": (
                "NoCutoff at frame-matched unwrapped conformations and original exceptions"
            ),
            "physical_accuracy_validated": False,
            "hsp_predictions": None,
        }

    def _unwrap(self, positions, lengths):
        result = positions.copy()
        for group in self._groups:
            reached, pending = {group[0]}, [group[0]]
            while pending:
                a = pending.pop()
                for b in self._neighbors[a]:
                    displacement = positions[b] - positions[a]
                    displacement -= lengths * np.rint(displacement / lengths)
                    candidate = result[a] + displacement
                    if b in reached:
                        if not np.allclose(result[b], candidate, atol=1e-5, rtol=0):
                            raise ValueError(
                                "Cannot unwrap a periodic winding or inconsistent cycle"
                            )
                    else:
                        result[b] = candidate
                        reached.add(b)
                        pending.append(b)
        return result

    def snapshot(self, positions_nm, box_vectors_nm):
        if self._closed:
            raise ValueError("Observer is closed")
        positions, box = (
            np.asarray(positions_nm, dtype=float),
            np.asarray(box_vectors_nm, dtype=float),
        )
        if positions.shape != (self._n, 3) or not np.isfinite(positions).all():
            raise ValueError("One finite nm coordinate triplet per particle is required")
        if (
            box.shape != (3, 3)
            or not np.isfinite(box).all()
            or not np.allclose(box, np.diag(np.diag(box)), atol=1e-12, rtol=0)
            or (np.diag(box) <= 0).any()
        ):
            raise ValueError("A positive orthorhombic cell in nm is required")
        lengths = np.diag(box)
        if self._periodic and self._cutoff_nm > lengths.min() / 2:
            raise ValueError("PME cutoff exceeds half the shortest cell length")
        unwrapped = self._unwrap(positions, lengths) if self._periodic else positions
        self._bulk_context.setPeriodicBoxVectors(*box)
        self._bulk_context.setPositions(unwrapped)
        self._gas_context.setPositions(unwrapped)

        def energy(context, group):
            return (
                context.getState(getEnergy=True, groups={group})
                .getPotentialEnergy()
                .value_in_unit(self._unit.kilojoule_per_mole)
            )

        bulk_lj, bulk_coulomb, full = [energy(self._bulk_context, group) for group in range(3)]
        isolated_lj, isolated_coulomb = [energy(self._gas_context, group) for group in range(2)]
        values = np.array([bulk_lj, bulk_coulomb, full, isolated_lj, isolated_coulomb])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite energy; geometry or force model needs review")
        closure = bulk_lj + bulk_coulomb - full
        # Relative rounding error scales with the operands, including when they cancel.
        closure_tolerance = 1e-5 + 1e-7 * (abs(bulk_lj) + abs(bulk_coulomb))
        if abs(closure) > closure_tolerance:
            raise ValueError("Split bulk energy does not reproduce the original nonbonded energy")
        volume = float(np.prod(lengths))
        lj = energy_density_mpa(isolated_lj - bulk_lj, volume)
        coulomb = energy_density_mpa(isolated_coulomb - bulk_coulomb, volume)
        return CohesionFrame(
            volume,
            bulk_lj,
            bulk_coulomb,
            isolated_lj,
            isolated_coulomb,
            lj,
            coulomb,
            lj + coulomb,
            closure,
            closure_tolerance,
        )

    def close(self):
        if not self._closed:
            self._bulk_context = self._gas_context = None
            self._bulk_integrator = self._gas_integrator = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
