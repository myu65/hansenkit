import numpy as np
import pytest

from hansenkit.chemistry import normalize_smiles
from hansenkit.intensive import intensive_features
from hansenkit.schema import EOPODistribution, SurfactantInput
from hansenkit.surfactants import assemble_blocks, characterize_surfactant


def surf(eo=2, po=0, **extra):
    return SurfactantInput(
        series_id="test-nonionic-block",
        tail_smiles="*CCCCCCCCCCCC",
        head_smiles="*O",
        eo_po=EOPODistribution(
            eo_mean=eo,
            po_mean=po,
            sequence="block",
            distribution="monodisperse",
            block_order="eo_then_po",
            **extra,
        ),
    )


def test_fatty_alcohol_eo_attachment_has_no_peroxide_or_extra_carbon():
    assert assemble_blocks("*CC", "*O", 2, 0) == normalize_smiles("CCOCCOCCO")
    assert assemble_blocks("*CC", "*O", 0, 0) == normalize_smiles("CCO")


def test_long_blocks_match_explicit_structure_and_are_bounded():
    result = characterize_surfactant(surf(80, 80))
    direct = intensive_features(assemble_blocks("*CCCCCCCCCCCC", "*O", 80, 80))
    np.testing.assert_allclose(result["features"], direct.values, atol=1e-7, rtol=1e-9)
    huge = characterize_surfactant(surf(10**7, 10**7))
    assert np.isfinite(huge["features"]).all()
    assert huge["max_explicit_block_counts"] == 32 and huge["hsp_predictions"] is None
    assert huge["calculated_mn_g_mol"] > 10**8
    assert huge["eo_residue_mass_g_mol"] == pytest.approx(44.053)
    assert huge["po_residue_mass_g_mol"] == pytest.approx(58.080)


def test_joint_pmf_retains_zero_blocks_and_mass_moments():
    dist = EOPODistribution(
        eo_mean=1,
        po_mean=1,
        sequence="block",
        block_order="eo_then_po",
        distribution="empirical",
        joint_pmf=((0, 0, 0.5), (2, 2, 0.5)),
        eo_std=1,
        po_std=1,
    )
    record = SurfactantInput(series_id="pmf", tail_smiles="*CCCC", head_smiles="*O", eo_po=dist)
    result = characterize_surfactant(record)
    assert result["population_basis"].startswith("joint_pmf")
    assert result["group_coverage_fraction"] == 1
    assert result["calculated_mn_g_mol"] == pytest.approx(74.123 + 44.053 + 58.080)
    with pytest.raises(ValueError, match="standard deviation"):
        EOPODistribution(
            eo_mean=1,
            po_mean=1,
            distribution="empirical",
            joint_pmf=dist.joint_pmf,
            eo_std=5,
        )


def test_ambiguous_connectivity_sequence_and_ionic_state_do_not_get_guessed():
    record = surf(2, 2)
    with pytest.raises(ValueError, match="block_order"):
        characterize_surfactant(
            record.model_copy(
                update={"eo_po": record.eo_po.model_copy(update={"block_order": None})}
            )
        )
    with pytest.raises(ValueError, match="ports"):
        characterize_surfactant(record.model_copy(update={"tail_smiles": "CCCC"}))
    with pytest.raises(ValueError, match="Ionic"):
        characterize_surfactant(record.model_copy(update={"ionic": True}))
    with pytest.raises(ValueError, match="sequence information"):
        characterize_surfactant(
            record.model_copy(
                update={"eo_po": record.eo_po.model_copy(update={"sequence": "random"})}
            )
        )
