"""Explicit local preparation of computed solvent-interaction auxiliary targets."""

import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from rdkit import Chem

from .auxiliary import AuxiliaryManifest
from .chemistry import UnstableSmilesError, normalize_smiles
from .data import write_json
from .embargo import HoldoutEmbargo
from .provenance import file_hash
from .splitting import TautomerGroupingError


class _NonfiniteJSONReader:
    """Stream Python JSON's bare nonfinite numbers as missing, preserving strings.

    Only complete unquoted NaN/Infinity/-Infinity tokens at value boundaries become
    null. Carry bytes and quote/escape state span arbitrary source/read boundaries.
    A null target becomes NaN in the numeric matrix and is explicitly rejected.
    """

    _tokens = re.compile(rb'"|\\|-?Infinity|NaN')
    _boundaries = b" \t\r\n,:[]{}"

    def __init__(self, stream):
        self.stream = stream
        self.carry = b""
        self.output = bytearray()
        self.offset = 0
        self.quoted = False
        self.escaped_at = -1
        self.finished = False
        self.previous = None

    def read(self, size=-1):
        if size == 0:
            return b""
        while not self.finished and (size < 0 or len(self.output) < size):
            chunk = self.stream.read(65536)
            self.finished = not chunk
            data = self.carry + chunk
            # Twelve bytes preserve the longest token and its following boundary.
            end = len(data) if self.finished else max(0, len(data) - 12)
            last = 0
            for match in self._tokens.finditer(data):
                start, stop = match.span()
                if start >= end:
                    break
                token = match.group()
                position = self.offset + start
                if self.quoted:
                    if position == self.escaped_at:
                        self.escaped_at = -1
                    elif token == b"\\":
                        self.escaped_at = position + 1
                    elif token == b'"':
                        self.quoted = False
                elif token == b'"':
                    self.quoted = True
                elif token != b"\\":
                    before = data[start - 1] if start else self.previous
                    after = data[stop] if stop < len(data) else None
                    if (before is None or before in self._boundaries) and (
                        after is None or after in self._boundaries
                    ):
                        self.output.extend(data[last:start])
                        self.output.extend(b"null")
                        last = stop
                        # A complete token can extend into the carry region.
                        end = max(end, stop)
            self.output.extend(data[last:end])
            if end:
                self.previous = data[end - 1]
            self.offset += end
            self.carry = data[end:]
        count = len(self.output) if size < 0 else min(size, len(self.output))
        result = bytes(self.output[:count])
        del self.output[:count]
        return result


@contextmanager
def _json_stream(path):
    if Path(path).suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".json")]
            if len(members) != 1 or members[0].file_size > 64 * 1024**3:
                raise ValueError("Require one JSON member, at most 64 GiB; no extraction occurs")
            with archive.open(members[0]) as stream:
                yield stream
    else:
        with Path(path).open("rb") as stream:
            yield stream


def _column(path, name):
    try:
        import ijson
    except ImportError as exc:
        raise ImportError("Install hansenkit[solvation-data] to read SolQuest JSON") from exc
    try:
        with _json_stream(path) as stream:
            return next(ijson.items(_NonfiniteJSONReader(stream), name, use_float=True))
    except (StopIteration, ijson.JSONError) as exc:
        raise ValueError(f"Missing or invalid SolQuest {name} column") from exc


def prepare_solquest(path, output, review: AuxiliaryManifest, embargo: HoldoutEmbargo):
    """Prepare unfitted arrays; exported permissions still prohibit auxiliary fitting.

    Review binds the raw JSON/ZIP hash, explicit target names and units. The operator
    must reconcile holdout identities and issue a new approved prepared-data manifest
    before fitting. No download, encoder, scaler, calibration or HSP computation occurs.
    """
    path, output = Path(path), Path(output)
    if output.exists():
        raise ValueError("SolQuest output directory exists; choose a fresh path")
    review.authorize()
    if file_hash(path) != review.sha256:
        raise ValueError("SolQuest source checksum mismatch")
    if any(unit != "kcal/mol" for unit in review.target_units):
        raise ValueError("SolQuest solvation targets require kcal/mol units")
    names = review.target_names
    smiles, columns = _column(path, "SMILES"), _column(path, "SOLVATION")
    if not isinstance(smiles, list) or not isinstance(columns, dict):
        raise ValueError("SolQuest needs a SMILES list and named SOLVATION columns")
    if set(columns) != set(names) or any(
        not isinstance(columns[n], list) or len(columns[n]) != len(smiles) for n in names
    ):
        raise ValueError("SolQuest target columns/names do not align with SMILES")
    try:
        targets = np.asarray([columns[n] for n in names], dtype=float).T
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SolQuest target columns must contain scalar numbers or missing values"
        ) from exc
    if targets.shape != (len(smiles), len(names)):
        raise ValueError("SolQuest target columns must be one-dimensional")
    # Keep memory proportional to identities/targets, never conformer/ECFP arrays.
    del columns
    rejected = {}
    kept, source_rows, positions, conflicting = {}, {}, {}, set()
    for index, raw in enumerate(smiles):
        try:
            canonical = normalize_smiles(raw)
        except UnstableSmilesError:
            rejected[index] = "unstable_canonical_identity"
            continue
        except (TypeError, ValueError):
            rejected[index] = "invalid_structure"
            continue
        mol = Chem.MolFromSmiles(canonical)
        if len(Chem.GetMolFrags(mol)) != 1 or any(
            a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetAtomicNum() == 0
            for a in mol.GetAtoms()
        ):
            rejected[index] = "charged_disconnected_radical_or_dummy"
            continue
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            rejected[index] = "nonorganic_structure"
            continue
        try:
            allowed = embargo.allows_canonical(canonical)
        except TautomerGroupingError:
            rejected[index] = "incomplete_tautomer_grouping"
            continue
        if not allowed:
            rejected[index] = "reserved_identity_or_scaffold"
            continue
        if not np.isfinite(targets[index]).all():
            rejected[index] = "missing_or_nonfinite_targets"
            continue
        if canonical in kept:
            positions[canonical].append(index)
            rejected[index] = "canonical_duplicate_rows"
            if not np.allclose(kept[canonical], targets[index], rtol=0, atol=1e-8):
                conflicting.add(canonical)
        else:
            kept[canonical], source_rows[canonical] = targets[index], index
            positions[canonical] = [index]
    for canonical in conflicting:
        for index in positions[canonical]:
            rejected[index] = "conflicting_duplicate_targets"
        del kept[canonical], source_rows[canonical]
    identities = sorted(kept)
    if not identities:
        raise ValueError("No eligible SolQuest structures after quality and holdout filtering")
    embargo.assert_disjoint(identities)
    if len(rejected) + len(identities) != len(smiles):
        raise RuntimeError("SolQuest source-row accounting failed")
    output.mkdir(parents=True)
    arrays = output / "computed-solvation.npz"
    np.savez_compressed(
        arrays,
        smiles=np.asarray(identities),
        targets=np.stack([kept[s] for s in identities]),
        target_names=np.asarray(names),
    )
    manifest = review.model_copy(
        update={
            "sha256": file_hash(arrays),
            "source": f"{review.source}; prepared from raw SHA-256 {review.sha256}; at 300 K",
            "training_allowed": False,
            "derived_weights_allowed": False,
            "redistribution_allowed": False,
        }
    )
    report = {
        "source_sha256": review.sha256,
        "prepared_sha256": manifest.sha256,
        "source_rows": len(smiles),
        "prepared_unique_molecules": len(identities),
        "target_count": len(names),
        "prepared_target_values": len(identities) * len(names),
        "row_rejection_counts": dict(Counter(rejected.values())),
        "rejected_source_rows": [
            {"source_row": i, "reason": rejected[i]} for i in sorted(rejected)
        ],
        "conflicting_duplicate_identities_excluded": len(conflicting),
        "duplicate_tolerance_kcal_mol": 1e-8,
        "nonfinite_token_handling": "Bare NaN/Infinity/-Infinity read as missing; quoted strings "
        "preserved; incomplete numeric target rows excluded, never filled with zero",
        "source_row_indices": [source_rows[s] for s in identities],
        "holdout_embargo": embargo.report(),
        "temperature_k": 300.0,
        "units": "kcal/mol",
        "task": "computed_solvation_auxiliary",
        "hsp_training_rows": 0,
        "model_fitting_performed": False,
        "prepared_data_training_allowed": False,
        "independent_hsp_accuracy_validated": False,
        "interpretation": "Fixed named solvent target coordinates; no solvent HSP values used. "
        "Filtering known reservations does not certify unresolved holdout identities. "
        "Inspect method/outliers and reconcile identities before issuing fitting permissions.",
    }
    write_json(output / "manifest.json", manifest.model_dump())
    write_json(output / "preparation.json", report)
    return report
