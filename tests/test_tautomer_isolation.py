from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from hansenkit.chemistry import normalize_smiles
from hansenkit.embargo import HoldoutEmbargo
from hansenkit.evaluation import calibrate, evaluate
from hansenkit.inference import predict_record
from hansenkit.models import train_model
from hansenkit.splitting import (
    GROUPING_POLICY,
    TautomerGroupingError,
    grouping_keys,
    scaffold_key,
    tautomer_parent_from_canonical,
)


@pytest.mark.parametrize(
    "left,right",
    [("O=C1CCCCC1", "OC1=CCCCC1"), ("c1c[nH]nn1", "c1cn[nH]n1")],
)
def test_tautomer_aliases_group_and_embargo_without_rewriting_inputs(left, right):
    originals = [normalize_smiles(s) for s in (left, right)]
    assert originals[0] != originals[1]
    assert scaffold_key(left) != scaffold_key(right)
    groups = grouping_keys(originals)
    assert groups[0] == groups[1]
    for reserved, query in [(left, right), (right, left)]:
        assert not HoldoutEmbargo.from_smiles([reserved]).eligible([query])[0]
    assert [normalize_smiles(s) for s in (left, right)] == originals


def test_legacy_scaffold_links_and_tautomer_links_join_transitively():
    values = ["O=C1CCCCC1", "OC1=CCCCC1", "C1=CCCCC1"]
    # The third shares the old enol scaffold; adding a tautomer alias cannot sever it.
    assert scaffold_key(values[1]) == scaffold_key(values[2])
    groups = grouping_keys(values)
    assert len(set(groups)) == 1
    series_groups = grouping_keys(values, ["a", "b", "b"], "polymer_series")
    assert len(set(series_groups)) == 1
    assert grouping_keys(["CCO", "CCCC"])[0] == grouping_keys(["CCO", "CCCC"])[1]


def test_embargo_closes_scaffold_bridges_without_needing_bridge_rows():
    # Keto and bare alkene are connected by a possible enol. The bridge need not be in the data.
    embargo = HoldoutEmbargo.from_smiles(["C1=CCCCC1"])
    assert not embargo.eligible(["O=C1CCCCC1"])[0]
    assert grouping_keys(["C1=CCCCC1", "O=C1CCCCC1"]).tolist() == [0, 0]


def test_topological_coarsening_preserves_heteroatom_connectivity():
    from hansenkit.splitting import core_topology_from_canonical

    assert core_topology_from_canonical("c1ccccc1") == core_topology_from_canonical("C1CCCCC1")
    assert core_topology_from_canonical("C1CCCCC1") != core_topology_from_canonical("C1CCNCC1")


def test_direct_embargo_construction_does_not_bypass_tautomer_aliases():
    left, right = "O=C1CCCCC1", "OC1=CCCCC1"
    assert not HoldoutEmbargo(frozenset([left]), frozenset([scaffold_key(left)])).eligible([right])[
        0
    ]
    assert not HoldoutEmbargo(frozenset(), frozenset([scaffold_key(left)])).eligible([right])[0]


def test_tautomer_grouping_preserves_charge_and_heavy_isotopes():
    neutral = tautomer_parent_from_canonical(normalize_smiles("NCC=O"))
    charged = tautomer_parent_from_canonical(normalize_smiles("[NH3+]CC=O"))
    assert Chem.GetFormalCharge(Chem.MolFromSmiles(neutral)) == 0
    assert Chem.GetFormalCharge(Chem.MolFromSmiles(charged)) == 1
    isotope = tautomer_parent_from_canonical(normalize_smiles("[13CH3]C(=O)C"))
    assert [a.GetIsotope() for a in Chem.MolFromSmiles(isotope).GetAtoms()].count(13) == 1


def test_incomplete_enumeration_is_refused_before_any_parent_is_chosen(monkeypatch):
    from hansenkit import splitting

    class IncompleteEnumerator:
        def SetMaxTautomers(self, value):
            pass

        def SetMaxTransforms(self, value):
            pass

        def Enumerate(self, mol):
            return SimpleNamespace(status=SimpleNamespace(name="MaxTautomers"))

        def PickCanonical(self, result):
            raise AssertionError("Incomplete candidates must not be used")

    monkeypatch.setattr(splitting.rdMolStandardize, "TautomerEnumerator", IncompleteEnumerator)
    with pytest.raises(TautomerGroupingError, match="Incomplete"):
        tautomer_parent_from_canonical(normalize_smiles("O=C1CCCCCCC1"))


def one_row(data, smiles):
    return replace(
        data,
        sample_ids=("original-tautomer-case",),
        smiles=(smiles,),
        targets=np.array([[10.0, 2.0, 3.0]]),
        series=("",),
    )


@pytest.mark.parametrize("function", [calibrate, evaluate])
def test_train_tautomer_cannot_supply_calibration_or_evaluation(synthetic, function):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    model.train_smiles = ("O=C1CCCCC1",)
    with pytest.raises(ValueError, match="training families or tautomer identities"):
        function(model, one_row(data, "OC1=CCCCC1"), [0], [0])


def test_evaluation_cannot_reuse_a_calibration_tautomer(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    model.train_smiles = ("c1ccccc1",)
    model.calibration_info = {
        "split_strategy": "scaffold",
        "grouping_policy": GROUPING_POLICY,
        "calibration_smiles": ["O=C1CCCCCC1"],
        "calibration_families": [scaffold_key("O=C1CCCCCC1")],
    }
    with pytest.raises(ValueError, match="calibration families or tautomer identities"):
        evaluate(model, one_row(data, "OC1=CCCCCC1"), [0], [0])


def test_programmatic_training_cannot_claim_policy_with_split_tautomers(synthetic):
    _, data, _ = synthetic
    tiny = replace(
        data,
        sample_ids=("a", "b", "c"),
        smiles=("O=C1CCCCC1", "OC1=CCCCC1", "c1ccccc1"),
        targets=np.array([[10.0, 2.0, 3.0]] * 3),
        series=("",) * 3,
    )
    with pytest.raises(ValueError, match="Provided groups split"):
        train_model(tiny, np.array([0]), np.arange(3))


def test_obsolete_calibration_is_not_reported_as_a_current_interval(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    model.conformal_radius = np.ones(3)
    model.calibration_info = {"grouping_policy": "legacy-murcko"}
    result = predict_record(model, {"kind": "molecule", "smiles": model.train_smiles[0]})
    assert result["predictions"] is not None
    assert result["intervals"] is None
    assert "calibration_grouping_policy_obsolete" in result["reasons"]
