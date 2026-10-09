import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from hansenkit.data import load_dataset
from hansenkit.encoders import LocalEmbeddingEncoder, get_encoder
from hansenkit.provenance import file_hash
from hansenkit.splitting import grouping_keys, scaffold_key, split_dataset


def test_dataset_gate_rejects_pending_hash_mismatch_mixed_labels(synthetic, tmp_path):
    paths, dataset, _ = synthetic
    manifest = dataset.manifest.model_dump()
    local = tmp_path / "manifest.json"
    for change, message in [
        ({"rights_status": "pending"}, "not approved"),
        ({"training_allowed": False}, "permissions"),
        ({"sha256": "0" * 64}, "checksum"),
    ]:
        local.write_text(json.dumps(manifest | change))
        with pytest.raises(ValueError, match=message):
            load_dataset(paths[0], local)
    data = tmp_path / "data.csv"
    data.write_bytes(paths[0].read_bytes().replace(b",synthetic,", b",teacher_reproduction,", 1))
    local.write_text(json.dumps(manifest | {"sha256": file_hash(data)}))
    with pytest.raises(ValueError, match="Mixed"):
        load_dataset(data, local)


def test_named_external_data_remains_blocked(synthetic, tmp_path):
    paths, dataset, _ = synthetic
    path = tmp_path / "HSP_SMILES.csv"
    path.write_bytes(paths[0].read_bytes())
    manifest = tmp_path / "m.json"
    manifest.write_text(dataset.manifest.model_dump_json())
    with pytest.raises(ValueError, match="blocked"):
        load_dataset(path, manifest)


def test_label_kind_contract(synthetic):
    _, dataset, _ = synthetic
    with pytest.raises(ValueError, match="teacher_id"):
        type(dataset.manifest).model_validate(
            dataset.manifest.model_dump() | {"label_kind": "teacher_reproduction"}
        )
    with pytest.raises(ValueError, match="independent measurements"):
        type(dataset.manifest).model_validate(
            dataset.manifest.model_dump() | {"label_kind": "experimental"}
        )


def test_reserved_evaluation_role_blocks_training_even_if_permissions_are_misconfigured(synthetic):
    _, dataset, _ = synthetic
    manifest = type(dataset.manifest).model_validate(
        dataset.manifest.model_dump() | {"allowed_role": "evaluation_only"}
    )
    manifest.authorize("evaluate")
    with pytest.raises(ValueError, match="Evaluation-only"):
        manifest.authorize("train")


def test_scaffold_and_canonical_duplicate_isolation(synthetic):
    _, dataset, split = synthetic
    partitions = (split.train, split.calibration, split.test)
    for i, left in enumerate(partitions):
        for right in partitions[i + 1 :]:
            assert not set(split.groups[left]) & set(split.groups[right])
            assert not {scaffold_key(dataset.smiles[x]) for x in left} & {
                scaffold_key(dataset.smiles[x]) for x in right
            }
    assert scaffold_key("CCO") == scaffold_key("CCCC") == "ACYCLIC"
    groups = grouping_keys(["CCO", "OCC", "CCC", "CCCC"], ["a", "b", "b", "c"], "polymer_series")
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[2]


def test_series_split_transitive_duplicate_isolation(synthetic):
    _, dataset, _ = synthetic
    series = tuple(f"family-{i % 12}" for i in range(len(dataset.smiles)))
    data = replace(dataset, series=series)
    split = split_dataset(data, "polymer_series")
    assert not {series[i] for i in split.train} & {series[i] for i in split.test}
    with pytest.raises(ValueError, match="every row"):
        split_dataset(dataset, "polymer_series")


def test_molformer_disabled_and_local_embeddings_rights_checked(tmp_path):
    with pytest.raises(ValueError, match="disabled"):
        get_encoder("molformer")
    csv_path, manifest_path = tmp_path / "e.csv", tmp_path / "e.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([["smiles", "e0", "e1"], ["CCO", 0.2, 0.4]])
    manifest = {
        "encoder_id": "test-fixed-vectors",
        "revision": "original-v1",
        "sha256": file_hash(csv_path),
        "weights_license": "MIT",
        "code_license": "MIT",
        "dependency_review": "approved",
        "rights_status": "approved",
        "permission_evidence": "Original test-authored vectors",
        "frozen": True,
        "training_allowed": True,
        "derived_weights_allowed": True,
        "commercial_use_allowed": True,
    }
    manifest_path.write_text(json.dumps(manifest))
    encoder = LocalEmbeddingEncoder(csv_path, manifest_path)
    np.testing.assert_allclose(encoder.transform(["OCC"]), [[0.2, 0.4]])
    with pytest.raises(ValueError, match="Missing"):
        encoder.transform(["CCC"])
    for change in ({"weights_license": "CC-BY-NC-4.0"}, {"dependency_review": "pending"}):
        manifest_path.write_text(json.dumps(manifest | change))
        with pytest.raises(ValueError):
            LocalEmbeddingEncoder(csv_path, manifest_path)
