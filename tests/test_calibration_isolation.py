from dataclasses import replace

import numpy as np
import pytest

from hansenkit.evaluation import calibrate, evaluate
from hansenkit.models import HSPModel, train_model
from hansenkit.splitting import scaffold_key


def one_row(data, smiles, series=""):
    return replace(
        data,
        sample_ids=("original-isolation-case",),
        smiles=(smiles,),
        targets=np.array([[10.0, 2.0, 3.0]]),
        series=(series,),
    )


def substituted(smiles):
    # Replace a terminal H by methyl while preserving the Murcko family.
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() and not atom.GetIsAromatic():
            builder = Chem.RWMol(mol)
            idx = builder.AddAtom(Chem.Atom(6))
            builder.GetAtomWithIdx(idx).SetIsotope(13)
            builder.AddBond(atom.GetIdx(), idx, Chem.BondType.SINGLE)
            output = builder.GetMol()
            Chem.SanitizeMol(output)
            return Chem.MolToSmiles(output)
    raise AssertionError("Fixture needs a substitutable carbon")


@pytest.mark.parametrize("function", [calibrate, evaluate])
def test_distinct_training_scaffold_neighbor_is_rejected(synthetic, function):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    neighbor = substituted(model.train_smiles[0])
    assert scaffold_key(neighbor) == scaffold_key(model.train_smiles[0])
    assert neighbor not in model.train_smiles
    with pytest.raises(ValueError, match="training families"):
        function(model, one_row(data, neighbor), [0], np.array([0]))


def test_evaluation_only_labels_cannot_fit_calibration_scores(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    holdout = replace(
        data, manifest=data.manifest.model_copy(update={"allowed_role": "evaluation_only"})
    )
    with pytest.raises(ValueError, match="evaluation-only|Evaluation-only"):
        calibrate(model, holdout, split.calibration, split.groups)


def test_calibration_family_is_reserved_after_reload(synthetic, tmp_path):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    calibrate(model, data, split.calibration, split.groups)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = HSPModel.load(path)
    neighbor = substituted(data.smiles[split.calibration[0]])
    assert neighbor not in model.train_smiles
    assert scaffold_key(neighbor) in loaded.calibration_info["calibration_families"]
    with pytest.raises(ValueError, match="calibration families|calibration molecules"):
        evaluate(loaded, one_row(data, neighbor), [0], np.array([0]))


def test_series_policy_and_missing_series_do_not_fall_back_to_scaffolds(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    model.provenance.update(split_strategy="polymer_series", training_series=["series-A"])
    query = data.smiles[split.test[0]]
    with pytest.raises(ValueError, match="training families"):
        evaluate(model, one_row(data, query, "series-A"), [0], np.array([0]))
    with pytest.raises(ValueError, match="series metadata"):
        evaluate(model, one_row(data, query), [0], np.array([0]))


def test_noncanonical_training_alias_is_still_rejected(synthetic):
    from rdkit import Chem

    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    training_smiles = model.train_smiles[0]
    alias = Chem.MolToSmiles(Chem.MolFromSmiles(training_smiles), allHsExplicit=True)
    assert alias != training_smiles
    with pytest.raises(ValueError, match="training molecules"):
        evaluate(model, one_row(data, alias), [0], [0])


def test_callers_cannot_split_one_calibration_family_into_independent_scores(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    with pytest.raises(ValueError, match="split a canonical identity or declared family"):
        calibrate(model, data, split.calibration, np.arange(len(data.smiles)))


@pytest.mark.parametrize("indices", [[-1], [0, 0], [], [0.5]])
def test_invalid_partition_indices_are_rejected(synthetic, indices):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    with pytest.raises(ValueError, match="unique in-range"):
        evaluate(model, data, indices, split.groups)
