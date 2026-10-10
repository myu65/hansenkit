import io
import json
from zipfile import ZipFile

import numpy as np
import pytest

pytest.importorskip("ijson", reason="Install the optional solvation-data extra")

from hansenkit.auxiliary import AuxiliaryManifest
from hansenkit.embargo import HoldoutEmbargo
from hansenkit.provenance import file_hash
from hansenkit.solquest import _NonfiniteJSONReader, prepare_solquest


def source(tmp_path, structures, targets, zipped=False):
    payload = json.dumps({"ECFP": [[0] * 20], "SMILES": structures, "SOLVATION": targets})
    path = tmp_path / ("source.zip" if zipped else "source.json")
    if zipped:
        with ZipFile(path, "w") as archive:
            archive.writestr("nested/source.json", payload)
    else:
        path.write_text(payload, encoding="utf-8")
    review = AuxiliaryManifest(
        source="Original synthetic interaction arrays; not physical observations",
        sha256=file_hash(path),
        label_kind="quantum_computed_auxiliary",
        target_names=tuple(targets),
        target_units=("kcal/mol",) * len(targets),
        rights_status="approved",
        audit_basis="documented_permission",
        permission_evidence="Original authored tests",
        training_allowed=True,
        derived_weights_allowed=True,
    )
    return path, review


@pytest.mark.parametrize("zipped", [False, True])
def test_prepare_preserves_alignment_negative_values_and_pending_fit(tmp_path, zipped):
    path, review = source(
        tmp_path,
        ["C1CCCCC1", "C1CC1", "C1CC1", "CCO", "CCCO", "[Na+]", "C.C"],
        {"hexane": [-3, -1, -1, -7, -8, -20, 2], "h2o": [-4, -2, -2, -9, -10, -22, 3]},
        zipped,
    )
    out = tmp_path / "prepared"
    report = prepare_solquest(path, out, review, HoldoutEmbargo.from_smiles(["OCC"]))
    assert report["source_rows"] == 7
    assert report["prepared_unique_molecules"] == 2
    assert report["prepared_target_values"] == 4
    assert report["row_rejection_counts"]["reserved_identity_or_scaffold"] == 2
    assert report["model_fitting_performed"] is False
    assert report["hsp_training_rows"] == 0
    assert sum(report["row_rejection_counts"].values()) + 2 == report["source_rows"]
    assert len(report["rejected_source_rows"]) == 5
    with np.load(out / "computed-solvation.npz", allow_pickle=False) as arrays:
        result = dict(zip(arrays["smiles"], arrays["targets"], strict=True))
        np.testing.assert_array_equal(result["C1CC1"], [-1, -2])
        np.testing.assert_array_equal(result["C1CCCCC1"], [-3, -4])
        np.testing.assert_array_equal(arrays["target_names"], ["hexane", "h2o"])
    manifest = AuxiliaryManifest.model_validate_json((out / "manifest.json").read_text())
    assert manifest.sha256 == file_hash(out / "computed-solvation.npz")
    with pytest.raises(ValueError, match="permissions"):
        manifest.authorize()
    with pytest.raises(ValueError, match="exists"):
        prepare_solquest(path, out, review, HoldoutEmbargo.from_smiles([]))


def test_conflicting_duplicates_are_all_excluded_not_first_selected(tmp_path):
    path, review = source(tmp_path, ["C1CC1", "C1CC1", "C1CCC1"], {"h2o": [-1, -3, -2]})
    out = tmp_path / "prepared"
    report = prepare_solquest(path, out, review, HoldoutEmbargo.from_smiles([]))
    assert report["conflicting_duplicate_identities_excluded"] == 1
    assert report["row_rejection_counts"]["conflicting_duplicate_targets"] == 2
    assert report["rejected_source_rows"] == [
        {"source_row": 0, "reason": "conflicting_duplicate_targets"},
        {"source_row": 1, "reason": "conflicting_duplicate_targets"},
    ]
    with np.load(out / "computed-solvation.npz") as arrays:
        assert arrays["smiles"].tolist() == ["C1CCC1"]


@pytest.mark.parametrize("targets", [{"h2o": [1]}, {"h2o": [[1], [2]]}, {"other": [1, 2]}])
def test_malformed_or_misnamed_target_columns_refused(tmp_path, targets):
    path, review = source(tmp_path, ["C1CC1", "C1CCC1"], targets)
    if "other" in targets:
        review = review.model_copy(update={"target_names": ("h2o",)})
    with pytest.raises(ValueError, match="columns|names"):
        prepare_solquest(path, tmp_path / "prepared", review, HoldoutEmbargo.from_smiles([]))
    assert not (tmp_path / "prepared").exists()


def test_missing_nonfinite_and_invalid_structures_not_filled(tmp_path):
    path, review = source(
        tmp_path,
        ["C1CC1", "C1CCC1", "bad_smiles", "O", "C1CCCC1", "*C"],
        {"h2o": [None, "NaN", 1, -1, -2, 1]},
    )
    report = prepare_solquest(path, tmp_path / "prepared", review, HoldoutEmbargo.from_smiles([]))
    assert report["prepared_unique_molecules"] == 1
    assert report["row_rejection_counts"]["missing_or_nonfinite_targets"] == 2
    assert report["row_rejection_counts"]["invalid_structure"] == 1
    assert report["row_rejection_counts"]["nonorganic_structure"] == 1


def test_source_hash_rights_and_units_checked_before_reading(tmp_path):
    path, review = source(tmp_path, ["C1CC1"], {"h2o": [-1]})
    for update, message in (
        ({"sha256": "0" * 64}, "checksum"),
        ({"training_allowed": False}, "permissions"),
        ({"target_units": ("MPa^0.5",)}, "units"),
    ):
        with pytest.raises(ValueError, match=message):
            prepare_solquest(
                path,
                tmp_path / "prepared",
                review.model_copy(update=update),
                HoldoutEmbargo.from_smiles([]),
            )


def test_multiple_zip_members_and_incomplete_json_refused(tmp_path):
    path, review = source(tmp_path, ["C1CC1"], {"h2o": [-1]}, zipped=True)
    with ZipFile(path, "a") as archive:
        archive.writestr("other.json", "{}")
    review = review.model_copy(update={"sha256": file_hash(path)})
    with pytest.raises(ValueError, match="one JSON"):
        prepare_solquest(path, tmp_path / "prepared", review, HoldoutEmbargo.from_smiles([]))
    path = tmp_path / "broken.json"
    path.write_text('{"SMILES": ["C1CC1"], "SOLVATION":', encoding="utf-8")
    review = review.model_copy(update={"sha256": file_hash(path)})
    with pytest.raises(ValueError, match="Missing or invalid"):
        prepare_solquest(path, tmp_path / "prepared", review, HoldoutEmbargo.from_smiles([]))


def test_cli_requires_explicit_reservations_and_outputs_pending_manifest(tmp_path, capsys):
    from hansenkit.cli import main

    path, review = source(tmp_path, ["C1CC1", "CCO"], {"h2o": [-1, -2]})
    review_path = tmp_path / "review.json"
    review_path.write_text(review.model_dump_json(), encoding="utf-8")
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"smiles": ["CCCO"]}), encoding="utf-8")
    output = tmp_path / "prepared"
    argv = [
        "solquest-prepare",
        "--source",
        str(path),
        "--source-review",
        str(review_path),
        "--reserved-identities",
        str(reserved),
        "--out",
        str(output),
    ]
    main(argv)
    result = json.loads(capsys.readouterr().out)
    assert result["prepared_unique_molecules"] == 1
    assert result["prepared_data_training_allowed"] is False
    assert "source_row_indices" not in result
    assert "rejected_source_rows" not in result
    saved = json.loads((output / "preparation.json").read_text())
    assert saved["reserved_identities_sha256"] == file_hash(reserved)
    reserved.write_text(json.dumps({"smiles": []}), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(argv[:-1] + [str(tmp_path / "not-created")])
    assert error.value.code == 2


@pytest.mark.parametrize("chunk_size", [1, 3, 13, 65536])
def test_nonfinite_reader_preserves_strings_and_chunk_boundary_escapes(chunk_size):
    class ShortReads(io.BytesIO):
        def read(self, n=-1):
            return super().read(min(n, chunk_size) if n >= 0 else chunk_size)

    source = json.dumps(
        {
            "NaN": 'NaN Infinity -Infinity "x" \\ end',
            "values": [float("nan"), float("inf"), -float("inf"), -1.3],
        }
    ).encode()
    reader = _NonfiniteJSONReader(ShortReads(source))
    pieces = []
    while piece := reader.read(7):
        pieces.append(piece)
    result = json.loads(b"".join(pieces))
    assert result["NaN"] == 'NaN Infinity -Infinity "x" \\ end'
    assert result["values"] == [None, None, None, -1.3]


def test_real_format_bare_nan_targets_excluded(tmp_path):
    path, review = source(tmp_path, ["C1CC1", "C1CCC1"], {"h2o": [float("nan"), -2]})
    report = prepare_solquest(path, tmp_path / "prepared", review, HoldoutEmbargo.from_smiles([]))
    assert report["prepared_unique_molecules"] == 1
    assert report["row_rejection_counts"]["missing_or_nonfinite_targets"] == 1
