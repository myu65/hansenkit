import json

import numpy as np
import pytest

from hansenkit.auxiliary import AuxiliaryManifest, fit_auxiliary, save_auxiliary
from hansenkit.embargo import HoldoutEmbargo
from hansenkit.encoders import MorganEncoder
from hansenkit.provenance import file_hash
from hansenkit.splitting import GROUPING_POLICY


def test_holdout_blocks_canonical_scaffold_and_series():
    embargo = HoldoutEmbargo.from_smiles(["OCC", "c1ccccc1"], ["peg"])
    # Exact ethanol, another acyclic compound and benzene analogs are all reserved.
    assert embargo.eligible(["CCO", "CCCC", "Oc1ccccc1", "C1CCC1"]).tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert not embargo.eligible(["C1CCC1"], ["peg"]).any()
    with pytest.raises(ValueError, match="Reserved holdout"):
        embargo.assert_disjoint(["CCCO"])


def test_auxiliary_embargo_runs_before_encoder_or_fit(tmp_path):
    path = tmp_path / "qm.npz"
    np.savez(path, smiles=np.array(["CCO"] * 12), targets=np.ones((12, 1)), target_names=["mu"])
    manifest = AuxiliaryManifest(
        source="Original test arrays",
        sha256=file_hash(path),
        label_kind="quantum_computed_auxiliary",
        target_names=("mu",),
        target_units=("Debye",),
        rights_status="approved",
        audit_basis="documented_permission",
        permission_evidence="test",
        training_allowed=True,
        derived_weights_allowed=True,
    )
    with pytest.raises(ValueError, match="Reserved holdout"):
        fit_auxiliary(path, manifest, MorganEncoder(), HoldoutEmbargo.from_smiles(["CCCO"]))
    with pytest.raises(ValueError, match="checksum"):
        fit_auxiliary(
            path,
            manifest.model_copy(update={"sha256": "0" * 64}),
            MorganEncoder(),
            HoldoutEmbargo.from_smiles([]),
        )


def test_auxiliary_does_not_masquerade_as_hsp():
    with pytest.raises(ValueError, match="HSP targets"):
        AuxiliaryManifest(
            source="test",
            sha256="0" * 64,
            label_kind="quantum_computed_auxiliary",
            target_names=("delta_d",),
            target_units=("MPa^0.5",),
            rights_status="approved",
            audit_basis="documented_permission",
            permission_evidence="test",
        )


def test_auxiliary_fit_round_trip_and_scaffold_counts(synthetic, tmp_path):
    from hansenkit.models import LinearHead

    _, data, _ = synthetic
    path = tmp_path / "quantum-test.npz"
    targets = np.arange(len(data.smiles), dtype=float)[:, None] / 100
    np.savez(path, smiles=np.array(data.smiles), targets=targets, target_names=["mu"])
    manifest = AuxiliaryManifest(
        source="Original arithmetic test arrays, no physical validity claimed",
        sha256=file_hash(path),
        label_kind="quantum_computed_auxiliary",
        target_names=("mu",),
        target_units=("Debye",),
        rights_status="approved",
        audit_basis="documented_permission",
        permission_evidence="Original test fixtures",
        training_allowed=True,
        derived_weights_allowed=True,
    )
    encoder = MorganEncoder()
    state, report = fit_auxiliary(path, manifest, encoder, HoldoutEmbargo.from_smiles([]))
    assert report["scaffold_overlap"] == 0 and report["hsp_training_rows"] == 0
    assert report["quantum_training_rows"] + report["quantum_test_rows"] == len(data.smiles)
    assert state["grouping_policy"] == report["grouping_policy"] == GROUPING_POLICY
    model_path = tmp_path / "auxiliary.json"
    save_auxiliary(state, model_path)
    restored = json.loads(model_path.read_text(encoding="utf-8"))
    assert restored["grouping_policy"] == restored["report"]["grouping_policy"] == GROUPING_POLICY
    head = LinearHead.restore(state["head"])
    x = encoder.transform(data.smiles[:5])
    assert np.isfinite(head.predict(x)).all()
    np.testing.assert_array_equal(head.predict(x), LinearHead.restore(head.state()).predict(x))
