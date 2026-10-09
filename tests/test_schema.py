import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hansenkit.schema import INPUT_ADAPTER, EOPODistribution, PolymerInput, RepeatUnit
from hansenkit.scope import assess_scope


def test_polymer_example_valid_but_no_prediction_scope():
    data = json.loads((Path(__file__).parents[1] / "examples/polymer.json").read_text())
    polymer = INPUT_ADAPTER.validate_python(data)
    assert polymer.mn_g_mol == 1000
    assert not assess_scope(polymer).supported


@pytest.mark.parametrize("smiles", ["CCO", "*CCO", "*C(*)O*", "*CC.*O", "**CC*"])
def test_repeat_unit_requires_two_connected_single_ports(smiles):
    with pytest.raises(ValidationError):
        RepeatUnit(smiles=smiles, mole_fraction=1)


def test_mass_fraction_and_extra_fields():
    with pytest.raises(ValidationError):
        PolymerInput(
            series_id="peg", repeat_units=[{"smiles": "*CCO*", "mole_fraction": 0.5}], mn_g_mol=1000
        )
    with pytest.raises(ValidationError):
        PolymerInput(
            series_id="peg", repeat_units=[{"smiles": "*CCO*", "mole_fraction": 1}], mn_g_mol=-1
        )
    with pytest.raises(ValidationError):
        INPUT_ADAPTER.validate_python({"kind": "molecule", "smiles": "CCO", "mn_g_mol": 10})


def test_empirical_distribution_validates_probability_and_means():
    dist = EOPODistribution(
        eo_mean=1, po_mean=1, distribution="empirical", joint_pmf=((0, 2, 0.5), (2, 0, 0.5))
    )
    assert dist.joint_pmf
    for bins in (((0, 2, 0.8), (2, 0, 0.8)), ((0, 2, 1),), ((0, 2, float("nan")),)):
        with pytest.raises(ValidationError):
            EOPODistribution(eo_mean=1, po_mean=1, distribution="empirical", joint_pmf=bins)
    with pytest.raises(ValidationError):
        EOPODistribution(eo_mean=1.5, po_mean=0, distribution="monodisperse")


def test_mixture_schema_does_not_enable_inference():
    mixture = INPUT_ADAPTER.validate_python(
        {
            "kind": "mixture",
            "components": [{"smiles": "CCO"}, {"smiles": "CCC"}],
            "mole_fractions": [0.5, 0.5],
        }
    )
    assert not assess_scope(mixture).supported
    with pytest.raises(ValidationError):
        INPUT_ADAPTER.validate_python(
            {
                "kind": "mixture",
                "components": [{"smiles": "CCO"}, {"smiles": "CCC"}],
                "mole_fractions": [0.3, 0.3],
            }
        )
