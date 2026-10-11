"""Analytic original fixtures; no external parameter table, molecule or weight."""

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from hansenkit.openmm_cohesion import (  # noqa: E402
    AVOGADRO,
    COULOMB_K,
    FrameCohesionObserver,
    energy_density_mpa,
)


def dimers(periodic=False):
    system, force, bond_force = mm.System(), mm.NonbondedForce(), mm.HarmonicBondForce()
    for charge in [0.1, -0.1, 0.1, -0.1]:
        system.addParticle(12)
        force.addParticle(charge, 0.2, 0.07)
    for a, b in [(0, 1), (2, 3)]:
        force.addException(a, b, -0.004, 0.12, 0.003)
        bond_force.addBond(a, b, 0.1, 200)
    if periodic:
        force.setNonbondedMethod(mm.NonbondedForce.PME)
        force.setCutoffDistance(0.9)
        force.setEwaldErrorTolerance(1e-6)
    else:
        force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    system.addForce(force)
    system.addForce(bond_force)
    return system


def test_density_conversion_uses_energy_of_cell_and_retains_sign():
    energy, volume = 602.214076, 1.0
    # Convert kJ/mol to J per cell, nm³ to m³, then Pa to MPa.
    independent = energy * 1000 / 6.02214076e23 / (volume * 1e-27) / 1e6
    assert energy_density_mpa(energy, volume) == pytest.approx(independent)
    assert energy_density_mpa(-energy, volume) == pytest.approx(-1000)
    assert AVOGADRO == 6.02214076e23


@pytest.mark.parametrize("energy,volume", [(np.nan, 1), (1, 0), (1, -1), (1, np.inf)])
def test_invalid_energy_or_volume_refused(energy, volume):
    with pytest.raises(ValueError):
        energy_density_mpa(energy, volume)


def test_pair_energies_and_original_intramolecular_exceptions():
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]])
    charges = [0.1, -0.1, 0.1, -0.1]
    cross_lj, cross_coulomb = 0.0, 0.0
    for a in [0, 1]:
        for b in [2, 3]:
            r = np.linalg.norm(positions[a] - positions[b])
            cross_lj += 4 * 0.07 * ((0.2 / r) ** 12 - (0.2 / r) ** 6)
            cross_coulomb += COULOMB_K * charges[a] * charges[b] / r
    isolated_lj = 2 * 4 * 0.003 * ((0.12 / 0.1) ** 12 - (0.12 / 0.1) ** 6)
    isolated_coulomb = 2 * COULOMB_K * -0.004 / 0.1
    system = dimers()
    before = mm.XmlSerializer.serialize(system)
    with FrameCohesionObserver(
        system, [0, 0, 1, 1], [(0, 1), (2, 3)], platform="Reference"
    ) as observer:
        result = observer.snapshot(positions, np.eye(3) * 3)
    assert result.isolated_lj_kj_mol == pytest.approx(isolated_lj, abs=1e-10)
    assert result.isolated_coulomb_kj_mol == pytest.approx(isolated_coulomb, abs=1e-10)
    assert result.bulk_lj_kj_mol - isolated_lj == pytest.approx(cross_lj, abs=1e-10)
    assert result.bulk_coulomb_kj_mol - isolated_coulomb == pytest.approx(cross_coulomb, abs=1e-9)
    assert result.ced_nonbonded_mpa == pytest.approx(
        energy_density_mpa(-cross_lj - cross_coulomb, 27), abs=1e-10
    )
    assert not result.physical_accuracy_validated
    assert not result.hsp_component_split_identified and result.hsp_predictions is None
    assert mm.XmlSerializer.serialize(system) == before
    with pytest.raises(ValueError, match="closed"):
        observer.snapshot(positions, np.eye(3) * 3)


def test_repulsive_frame_is_not_clipped_to_positive_cohesion():
    system, force = mm.System(), mm.NonbondedForce()
    for _ in range(2):
        system.addParticle(12)
        force.addParticle(0, 0.3, 0.1)
    system.addForce(force)
    with FrameCohesionObserver(system, [0, 1], [], platform="Reference") as observer:
        frame = observer.snapshot([[0, 0, 0], [0.25, 0, 0]], np.eye(3) * 3)
    assert frame.ced_lj_mpa < 0 and frame.ced_nonbonded_mpa < 0
    assert frame.hsp_predictions is None


def test_periodic_wrapping_retains_intramolecular_geometry():
    system = dimers(periodic=True)
    wrapped = np.array([[2.95, 0, 0], [0.05, 0, 0], [1.1, 0, 0], [1.2, 0, 0]])
    unwrapped = wrapped.copy()
    unwrapped[1, 0] = 3.05
    with FrameCohesionObserver(
        system, [0, 0, 1, 1], [(0, 1), (2, 3)], platform="Reference"
    ) as observer:
        first = observer.snapshot(wrapped, np.eye(3) * 3)
        second = observer.snapshot(unwrapped, np.eye(3) * 3)
        for name in ["bulk_lj_kj_mol", "bulk_coulomb_kj_mol", "isolated_coulomb_kj_mol"]:
            assert getattr(first, name) == pytest.approx(getattr(second, name), abs=1e-8)
        with pytest.raises(ValueError, match="cutoff"):
            observer.snapshot(wrapped, np.eye(3))
        with pytest.raises(ValueError, match="orthorhombic"):
            observer.snapshot(wrapped, [[3, 0, 0], [0.1, 3, 0], [0, 0, 3]])


def test_cpu_and_reference_agree_for_original_fixture():
    positions = [[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]]
    frames = []
    for platform in ["CPU", "Reference"]:
        with FrameCohesionObserver(
            dimers(), [0, 0, 1, 1], [(0, 1), (2, 3)], platform=platform, threads=1
        ) as observer:
            frames.append(observer.snapshot(positions, np.eye(3) * 3))
    assert frames[0].ced_nonbonded_mpa == pytest.approx(frames[1].ced_nonbonded_mpa, abs=1e-7)


@pytest.mark.parametrize("platform", ["CUDA", "OpenCL", "automatic"])
def test_observer_never_selects_gpu_implicitly(platform):
    with pytest.raises(ValueError, match="CPU or Reference"):
        FrameCohesionObserver(dimers(), [0, 0, 1, 1], [(0, 1), (2, 3)], platform=platform)


def test_incorrect_partition_or_charge_or_force_model_refused():
    with pytest.raises(ValueError, match="integer molecule"):
        FrameCohesionObserver(dimers(), [0.0, 0, 1, 1], [(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="one molecule"):
        FrameCohesionObserver(dimers(), [0, 1, 1, 0], [(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="omits"):
        FrameCohesionObserver(dimers(), [0, 0, 1, 1], [])
    system = dimers()
    system.getForce(0).setParticleParameters(0, 0.2, 0.2, 0.07)
    with pytest.raises(ValueError, match="neutral"):
        FrameCohesionObserver(system, [0, 0, 1, 1], [(0, 1), (2, 3)])
    system = dimers()
    system.addForce(mm.CustomExternalForce("0"))
    with pytest.raises(ValueError, match="profile"):
        FrameCohesionObserver(system, [0, 0, 1, 1], [(0, 1), (2, 3)])
    system = dimers()
    system.getForce(0).addGlobalParameter("lambda", 0)
    system.getForce(0).addParticleParameterOffset("lambda", 0, 0.1, 0, 0)
    with pytest.raises(ValueError, match="offset"):
        FrameCohesionObserver(system, [0, 0, 1, 1], [(0, 1), (2, 3)])
    system = dimers(periodic=True)
    system.getForce(0).setReciprocalSpaceForceGroup(5)
    with pytest.raises(ValueError, match="reciprocal"):
        FrameCohesionObserver(system, [0, 0, 1, 1], [(0, 1), (2, 3)])


def test_periodic_winding_cannot_be_silently_unwrapped():
    system, force, bonds = mm.System(), mm.NonbondedForce(), [(0, 1), (1, 2), (2, 0)]
    for _ in range(4):
        system.addParticle(12)
        force.addParticle(0, 0.2, 0.07)
    force.setNonbondedMethod(mm.NonbondedForce.PME)
    force.setCutoffDistance(0.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    system.addForce(force)
    with FrameCohesionObserver(system, [0, 0, 0, 1], bonds, platform="Reference") as observer:
        with pytest.raises(ValueError, match="winding"):
            observer.snapshot([[0.2, 0, 0], [1.2, 0, 0], [2.2, 0, 0], [0, 1, 0]], np.eye(3) * 3)


def test_original_no_cutoff_molecule_matches_isolated_energy():
    # An independent standard NonbondedForce evaluates one molecule, including its exception.
    single, force = mm.System(), mm.NonbondedForce()
    for charge in [0.1, -0.1]:
        single.addParticle(12)
        force.addParticle(charge, 0.2, 0.07)
    force.addException(0, 1, -0.004, 0.12, 0.003)
    single.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(single, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [0.1, 0, 0]])
    independent = (
        context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    del context, integrator
    with FrameCohesionObserver(
        dimers(), [0, 0, 1, 1], [(0, 1), (2, 3)], platform="Reference"
    ) as observer:
        result = observer.snapshot([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]], np.eye(3) * 3)
    assert result.isolated_lj_kj_mol + result.isolated_coulomb_kj_mol == pytest.approx(
        2 * independent, abs=1e-9
    )


def test_cancelling_components_allow_roundoff_but_reject_wrong_total():
    class EnergyState:
        def __init__(self, energy):
            self.energy = energy

        def getPotentialEnergy(self):
            return self.energy * unit.kilojoule_per_mole

    class EnergyContext:
        # Authored energy values exercise cancellation without importing pilot records.
        full = 400.00006

        def setPeriodicBoxVectors(self, *vectors):
            pass

        def setPositions(self, positions):
            pass

        def getState(self, *, getEnergy, groups):
            values = [-2000.0, 2400.0, self.full]
            return EnergyState(values[next(iter(groups))])

    with FrameCohesionObserver(
        dimers(), [0, 0, 1, 1], [(0, 1), (2, 3)], platform="Reference"
    ) as observer:
        context = EnergyContext()
        observer._bulk_context = context
        frame = observer.snapshot([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]], np.eye(3) * 3)
        assert (
            abs(frame.bulk_split_closure_error_kj_mol) < frame.bulk_split_closure_tolerance_kj_mol
        )
        assert frame.bulk_split_closure_tolerance_kj_mol < 0.0005
        context.full += 0.001
        with pytest.raises(ValueError, match="Split bulk energy"):
            observer.snapshot([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]], np.eye(3) * 3)
