import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from hansenkit.encoders import get_encoder
from hansenkit.molformer import (
    CheckpointReview,
    FrozenMolformerEncoder,
    _load_reviewed_source,
    fetch_checkpoint,
    load_encoder_tensors,
    nonisomeric_smiles,
    verify_checkpoint,
)


def approved_review():
    path = Path(__file__).parents[1] / "assets/molformer-review.json"
    return CheckpointReview.model_validate_json(path.read_text())


def test_unreviewed_revision_or_changed_code_is_never_authorized():
    review = approved_review()
    review.authorize()
    with pytest.raises(ValueError):
        CheckpointReview.model_validate(review.model_dump() | {"revision": "unreviewed"})
    changed = dict(review.files)
    changed["modeling_molformer.py"] = "0" * 64
    with pytest.raises(ValueError, match="audited publisher release"):
        review.model_copy(update={"files": changed}).authorize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("rights_status", "pending"),
        ("dependency_review", "pending"),
        ("reviewed_local_code_execution_allowed", False),
        ("derived_weights_allowed", False),
    ],
)
def test_checkpoint_permissions_fail_before_code_loading(field, value):
    with pytest.raises(ValueError):
        approved_review().model_copy(update={field: value}).authorize()


def test_missing_or_modified_file_is_refused_without_import(tmp_path):
    review = approved_review()
    with pytest.raises(ValueError, match="missing"):
        verify_checkpoint(tmp_path, review, check_runtime=False)
    (tmp_path / "README.md").write_text("changed publisher bytes")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(tmp_path, review, check_runtime=False)


def test_local_checkpoint_is_required_and_default_is_lightweight():
    assert get_encoder().metadata()["pretrained"] is False
    with pytest.raises(ValueError, match="disabled"):
        get_encoder("molformer")


def test_pretraining_preprocessing_removes_stereo_without_changing_molecular_identity():
    assert nonisomeric_smiles("C[C@H](O)F") == nonisomeric_smiles("C[C@@H](O)F")
    assert nonisomeric_smiles("OCC") == "CCO"


def test_cached_fixed_vectors_preserve_order_and_duplicates_without_loading_model():
    encoder = object.__new__(FrozenMolformerEncoder)
    encoder._cache = {"CCO": np.arange(768, dtype=float), "CCC": np.ones(768)}

    def forbidden_load():
        raise AssertionError("A cached fixed vector must not reload or redraw the encoder")

    encoder._load = forbidden_load
    output = encoder.transform(["CCC", "OCC", "CCO"])
    np.testing.assert_array_equal(output[0], np.ones(768))
    np.testing.assert_array_equal(output[1], output[2])
    assert output.shape == (3, 768)


def test_review_file_contains_no_local_paths_or_weights():
    state = approved_review().model_dump()
    json.dumps(state)
    assert set(state["files"]) == {
        "README.md",
        "config.json",
        "configuration_molformer.py",
        "modeling_molformer.py",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_fetch_is_explicit_and_bad_download_never_becomes_a_checkpoint(tmp_path, monkeypatch):
    review_path = Path(__file__).parents[1] / "assets/molformer-review.json"
    calls = []

    def bad_download(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b"tampered publisher bytes")

    monkeypatch.setattr("hansenkit.molformer.urllib.request.urlopen", bad_download)
    with pytest.raises(ValueError, match="checksum"):
        fetch_checkpoint(tmp_path / "checkpoint", review_path)
    assert len(calls) == 1
    assert not (tmp_path / "checkpoint/README.md").exists()
    assert all(url.startswith("https://huggingface.co/ibm-research/") for url in calls)


def test_strict_tensor_loading_restores_pretrained_values(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    expected = {
        "molformer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "molformer.bias": torch.tensor([2.0, 3.0]),
    }
    path = tmp_path / "tiny.safetensors"
    safetensors.save_file(expected, path)
    restored = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        model = torch.nn.Linear(3, 2)
        load_encoder_tensors(model, path)
        restored.append(model.state_dict())
        assert torch.equal(model.weight, expected["molformer.weight"])
        assert torch.equal(model.bias, expected["molformer.bias"])
    assert torch.equal(restored[0]["weight"], restored[1]["weight"])
    bad = tmp_path / "incomplete.safetensors"
    safetensors.save_file({"molformer.weight": expected["molformer.weight"]}, bad)
    with pytest.raises(ValueError, match="architecture"):
        load_encoder_tensors(torch.nn.Linear(3, 2), bad)


def test_changed_reviewed_source_is_refused_before_execution(tmp_path):
    path = tmp_path / "reviewed.py"
    source = b"VALUE = 17\n"
    path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    module = _load_reviewed_source("_hansenkit_test_reviewed_source", path, digest)
    assert module.VALUE == 17
    path.write_text("raise AssertionError('changed source must never execute')")
    with pytest.raises(ValueError, match="changed before execution"):
        _load_reviewed_source("_hansenkit_test_unverified_source", path, digest)
