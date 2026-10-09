import csv
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from hansenkit.evaluation import calibrate, component_metrics, evaluate
from hansenkit.inference import predict_record
from hansenkit.models import HSPModel, train_model


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_fit_calibration_roundtrip_no_leakage(synthetic, tmp_path, mode):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups, mode)
    cal = tuple(data.smiles[i] for i in split.calibration)
    before = model.predict(cal)
    calibrate(model, data, split.calibration, split.groups)
    path = tmp_path / f"{mode}.json"
    model.save(path)
    loaded = HSPModel.load(path)
    np.testing.assert_allclose(before, loaded.predict(cal))
    assert loaded.provenance["label_kind"] == "synthetic"
    assert loaded.provenance["residual_cross_fitted"] == (mode == "C")
    assert set(loaded.train_smiles).isdisjoint(cal)
    report = evaluate(loaded, data, split.test, split.groups)
    assert report["metrics"]["rows"] == len(split.test)
    assert report["real_accuracy_validated"] is False
    assert all(v["rmse"] >= v["mae"] for v in report["metrics"]["components"].values())


def test_test_labels_do_not_affect_model_or_intervals(synthetic):
    _, data, split = synthetic
    modified_targets = data.targets.copy()
    modified_targets[split.test] += 1000
    altered = replace(data, targets=modified_targets)
    models = [train_model(d, split.train, split.groups, "C") for d in (data, altered)]
    for model, d in zip(models, (data, altered), strict=True):
        calibrate(model, d, split.calibration, split.groups)
    np.testing.assert_array_equal(models[0].head.coef, models[1].head.coef)
    np.testing.assert_array_equal(models[0].residual.coef, models[1].residual.coef)
    np.testing.assert_array_equal(models[0].conformal_radius, models[1].conformal_radius)


def test_small_calibration_does_not_invent_finite_interval(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    calibrate(model, data, split.calibration, split.groups, alpha=0.05)
    assert model.conformal_radius is None
    assert not model.calibration_info["finite_interval_available"]
    with pytest.raises(ValueError, match="training molecules"):
        calibrate(model, data, split.train, split.groups)


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_low_level_model_also_refuses_unsupported_chemistry(synthetic, mode):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups, mode)
    with pytest.raises(ValueError, match="Unsupported"):
        model.predict(["*CC*"])
    with pytest.raises(ValueError, match="Unsupported"):
        model.predict(["C[n+]1ccccc1"])


def test_inference_refusals_and_provenance(synthetic):
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups)
    result = predict_record(model, {"kind": "molecule", "smiles": model.train_smiles[0]})
    assert result["status"] == "synthetic_demo"
    assert result["real_accuracy_validated"] is False
    for smiles in ("C[N+](C)(C)C", "CCO.CCC", "*CCO*", "CCCCCCCCOCCOCCOCCO"):
        result = predict_record(model, {"kind": "molecule", "smiles": smiles})
        assert result["status"] == "unsupported"
        assert result["predictions"] is None
    result = predict_record(model, {"kind": "molecule", "smiles": "C(F)(F)(F)C(F)(F)F"})
    assert result["status"] == "out_of_domain"
    assert result["predictions"] is None


def test_teacher_and_real_evaluation_are_separate(synthetic):
    _, data, split = synthetic
    teacher = data.manifest.model_copy(
        update={"label_kind": "teacher_reproduction", "teacher_id": "original-fake-teacher"}
    )
    teacher_data = replace(data, manifest=teacher)
    model = train_model(teacher_data, split.train, split.groups)
    reproduction = evaluate(model, teacher_data, split.test, split.groups)
    experimental = data.manifest.model_copy(
        update={"label_kind": "experimental", "independent_measurements": True}
    )
    real = evaluate(model, replace(data, manifest=experimental), split.test, split.groups)
    assert reproduction["evaluation_basis"] == "teacher_reproduction"
    assert real["evaluation_basis"] == "independent_measurements"
    assert real["training_label_kind"] == "teacher_reproduction"
    # These are test-only manifests, not claims that the fixture is actual experimental data.
    with pytest.raises(ValueError, match="training molecules"):
        evaluate(model, teacher_data, split.train, split.groups)


def test_metrics_constant_and_empty_are_json_safe():
    result = component_metrics(np.ones((3, 3)), np.zeros((3, 3)))
    assert result["components"]["delta_d"]["r2"] is None
    assert component_metrics(np.zeros((0, 3)), np.zeros((0, 3)))["components"] is None
    json.dumps(result, allow_nan=False)


def test_optional_lightgbm_roundtrip(synthetic, tmp_path):
    pytest.importorskip("lightgbm")
    _, data, split = synthetic
    model = train_model(data, split.train, split.groups, head="lightgbm")
    path = tmp_path / "lgb.json"
    model.save(path)
    query = tuple(data.smiles[i] for i in split.test)
    np.testing.assert_allclose(model.predict(query), HSPModel.load(path).predict(query))


def test_cli_csv_end_to_end(tmp_path):
    def cli(*arguments):
        subprocess.run(
            [sys.executable, "-m", "hansenkit.cli", *map(str, arguments)],
            check=True,
            capture_output=True,
            text=True,
        )

    synthetic_dir, models_dir = tmp_path / "syn", tmp_path / "models"
    cli("synthetic", "--out", synthetic_dir)
    cli(
        "compare",
        "--data",
        synthetic_dir / "synthetic.csv",
        "--manifest",
        synthetic_dir / "manifest.json",
        "--out",
        models_dir,
    )
    comparison = json.loads((models_dir / "comparison.json").read_text())
    assert len(comparison) == 3
    assert (
        comparison["A-ridge"]["split"]
        == comparison["B-ridge"]["split"]
        == comparison["C-ridge"]["split"]
    )
    input_path, output_path = tmp_path / "input.csv", tmp_path / "out.csv"
    input_path.write_text("sample_id,smiles\nneutral,CCO\nionic,C[N+](C)(C)C\nbad,invalid\n")
    cli(
        "predict",
        "--model",
        models_dir / "A-ridge.json",
        "--input",
        input_path,
        "--output",
        output_path,
    )
    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[1]["status"] == "unsupported" and rows[1]["delta_d"] == ""
    assert rows[2]["status"] == "invalid_input" and rows[2]["delta_h"] == ""
    assert all(row["label_kind"] == "synthetic" for row in rows)
