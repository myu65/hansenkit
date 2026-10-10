import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chemistry import FEATURE_NAMES, chemical_features, normalize_smiles
from .provenance import BLOCKED_NAMES, UNITS, DatasetManifest, file_hash
from .schema import MoleculeInput
from .scope import assess_scope

TARGETS = ("delta_d", "delta_p", "delta_h")
CSV_FIELDS = ("sample_id", "smiles", *TARGETS, "label_kind", "temperature_k", "polymer_series")


@dataclass(frozen=True)
class Dataset:
    sample_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    targets: np.ndarray
    series: tuple[str, ...]
    manifest: DatasetManifest


def load_dataset(path: str | Path, manifest_path: str | Path, purpose="train") -> Dataset:
    path = Path(path)
    manifest = DatasetManifest.model_validate_json(Path(manifest_path).read_text(encoding="utf-8"))
    manifest.authorize(purpose)
    if path.name.lower() in BLOCKED_NAMES and not manifest.restricted_asset_permission_evidence:
        raise ValueError("This named external dataset is blocked pending project rights review")
    if file_hash(path) != manifest.sha256:
        raise ValueError("Dataset checksum mismatch")
    if abs(manifest.temperature_k - 298.15) > 1e-6:
        raise ValueError("Temperature outside initial scope")
    ids, smiles, targets, series = [], [], [], []
    seen_ids = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "smiles", *TARGETS, "label_kind"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required columns")
        for index, row in enumerate(reader, start=2):
            if row["label_kind"] != manifest.label_kind:
                raise ValueError(f"Mixed or misdeclared label kind at row {index}")
            if row.get("kind", "molecule") != "molecule":
                raise ValueError(f"Non-molecule training is not validated (row {index})")
            record = MoleculeInput(
                smiles=row["smiles"],
                temperature_k=float(row.get("temperature_k") or manifest.temperature_k),
            )
            if not assess_scope(record).supported:
                raise ValueError(f"Unsupported training structure at row {index}")
            values = np.array([float(row[t]) for t in TARGETS])
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Targets must be finite and nonnegative at row {index}")
            if not row["sample_id"].strip() or row["sample_id"] in seen_ids:
                raise ValueError("sample_id must be nonempty and unique")
            ids.append(row["sample_id"])
            seen_ids.add(row["sample_id"])
            smiles.append(record.smiles)
            targets.append(values)
            series.append(row.get("polymer_series", ""))
    if len(ids) < 12:
        raise ValueError("At least 12 rows and sufficient distinct groups are required")
    return Dataset(tuple(ids), tuple(smiles), np.stack(targets), tuple(series), manifest)


def synthetic_manifest(path: Path, seed: int) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=f"hansenkit-original-synthetic-v1-seed-{seed}",
        sha256=file_hash(path),
        label_kind="synthetic",
        units=UNITS,
        source="Original project-authored SMILES and arbitrary synthetic formula; no HSP table",
        license="MIT",
        rights_status="approved",
        permission_evidence="Project LICENSE; generator in hansenkit.data (not empirical HSP)",
        approved_by="hansenkit synthetic generator",
        local_evaluation_allowed=True,
        training_allowed=True,
        derived_labels_allowed=True,
        derived_weights_allowed=True,
        redistribution_allowed=True,
    )


def generate_synthetic(directory: str | Path, seed: int = 42) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, manifest_path = directory / "synthetic.csv", directory / "manifest.json"
    if path.exists() or manifest_path.exists():
        raise ValueError("Synthetic output exists; choose a fresh directory")
    # Authored structure examples, not extracted from any HSP dataset.
    cores = (
        "c1ccccc1",
        "c1ccncc1",
        "c1ccoc1",
        "c1ccsc1",
        "c1cc[nH]c1",
        "C1CCCCC1",
        "C1CCCC1",
        "C1CCC1",
        "C1CC1",
        "C1CCCCCC1",
        "C1CCCCCCC1",
        "C1CCOCC1",
        "C1CCNCC1",
        "C1COCC1",
        "C1CNCC1",
        "C1COC1",
        "C1CNC1",
        "c1ccc2ccccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccc2occc2c1",
        "c1ccc2sccc2c1",
        "c1ccc(-c2ccccc2)cc1",
        "C1CCC2CCCCC2C1",
        "C1CC2CCC1C2",
        "c1ccc2[nH]ccc2c1",
        "c1cncnc1",
        "c1nccnc1",
        "C1CCSCC1",
    )
    prefixes = ("", "C", "CC", "CCC", "O", "N", "CO", "CC(=O)", "O=C(O)", "Cl", "CS")
    structures = {normalize_smiles(prefix + core) for core in cores for prefix in prefixes}
    for length in range(1, 9):
        for suffix in ("", "O", "N", "C(=O)O", "C#N", "OC"):
            structures.add(normalize_smiles("C" * length + suffix))
    rng = np.random.default_rng(seed)
    rows = []
    for smiles in sorted(structures):
        if not assess_scope(MoleculeInput(smiles=smiles)).supported:
            continue
        x = dict(zip(FEATURE_NAMES, chemical_features(smiles), strict=True))
        # These arbitrary numbers have no physical calibration or literature provenance.
        labels = np.array(
            [
                10 + 0.012 * x["molecular_weight"] + 0.17 * x["aromatic_c"] + 0.3 * x["rings"],
                1
                + 0.065 * x["tpsa"]
                + 0.25 * x["hbond_acceptors"]
                + 0.4 * np.cos(x["molecular_weight"] / 30),
                0.5
                + 0.9 * x["hbond_donors"]
                + 0.035 * x["tpsa"]
                + 0.25 * np.sin(x["rotatable_bonds"]),
            ]
        ) + rng.normal(0, 0.08, size=3)
        rows.append(
            {
                "sample_id": f"syn-{len(rows):04d}",
                "smiles": smiles,
                **dict(zip(TARGETS, labels, strict=True)),
                "label_kind": "synthetic",
                "temperature_k": 298.15,
                "polymer_series": "",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path.write_text(
        synthetic_manifest(path, seed).model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return path, manifest_path


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
