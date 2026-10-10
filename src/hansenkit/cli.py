import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .chemistry import FEATURE_NAMES, atom_coverage, chemical_features
from .data import TARGETS, generate_synthetic, load_dataset, write_json
from .encoders import get_encoder
from .evaluation import calibrate, evaluate
from .inference import predict_record
from .models import HSPModel, train_model
from .schema import INPUT_ADAPTER
from .scope import assess_scope
from .splitting import scaffold_key, split_dataset


def add_encoder(parser):
    parser.add_argument("--encoder", choices=("morgan", "local", "molformer"), default="morgan")
    parser.add_argument("--embedding-csv")
    parser.add_argument("--embedding-manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-review")
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")


def add_training(parser):
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("scaffold", "polymer_series"), default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.2)
    add_encoder(parser)


def configured_encoder(args):
    return get_encoder(
        args.encoder,
        args.embedding_csv,
        args.embedding_manifest,
        args.checkpoint,
        args.checkpoint_review,
        args.encoder_batch_size,
        args.encoder_device,
    )


def fit_and_report(dataset, split, args, mode, head="ridge", encoder=None):
    encoder = encoder or configured_encoder(args)
    model = train_model(
        dataset,
        split.train,
        split.groups,
        mode,
        encoder,
        head,
        args.seed,
        split_strategy=args.split,
    )
    model.provenance.update(
        {
            "split_strategy": args.split,
            "training_series": sorted(
                {dataset.series[i] for i in split.train if dataset.series[i]}
            ),
        }
    )
    calibrate(model, dataset, split.calibration, split.groups, encoder, args.alpha)
    report = evaluate(model, dataset, split.test, split.groups, encoder)
    report.update(
        {
            "mode": mode,
            "head": head,
            "encoder": encoder.metadata(),
            "seed": args.seed,
            "split_strategy": args.split,
            "split": split.report(dataset),
            "dataset_sha256": dataset.manifest.sha256,
        }
    )
    return model, report


def predict_csv(args):
    model, encoder = HSPModel.load(args.model), configured_encoder(args)
    path = Path(args.output)
    if path.exists():
        raise ValueError("Prediction output exists; choose a fresh output file")
    output = []
    with Path(args.input).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"smiles", "sample_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Prediction CSV needs sample_id and smiles columns")
        for row in reader:
            try:
                record = (
                    json.loads(row["record_json"])
                    if row.get("record_json")
                    else {
                        "kind": row.get("kind") or "molecule",
                        "smiles": row["smiles"],
                        "temperature_k": float(row.get("temperature_k") or 298.15),
                    }
                )
                result = predict_record(model, record, encoder)
            except (ValueError, KeyError):
                result = {
                    "status": "invalid_input",
                    "reasons": ["invalid_schema_or_missing_embedding"],
                    "label_kind": model.provenance["label_kind"],
                    "units": model.provenance["units"],
                    "predictions": None,
                }
            flat = {
                "sample_id": row["sample_id"],
                "smiles": row["smiles"],
                "canonical_smiles": result.get("canonical_smiles", ""),
                "status": result["status"],
                "reasons": "|".join(result["reasons"]),
                "label_kind": result["label_kind"],
                "units": result["units"],
                "real_accuracy_validated": False,
                "ood": result.get("ood", ""),
                "extrapolation": result.get("extrapolation", ""),
                "nearest_training_tanimoto": result.get("nearest_training_tanimoto", ""),
            }
            for target in TARGETS:
                flat[target] = (result.get("predictions") or {}).get(target, "")
                interval = (result.get("intervals") or {}).get(target, ("", ""))
                flat[f"{target}_lower"], flat[f"{target}_upper"] = interval
            output.append(flat)
    if not output:
        raise ValueError("Prediction CSV is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"rows": len(output), "output": str(path), "real_accuracy_validated": False}))


def external_evaluation(args):
    model = HSPModel.load(args.model)
    dataset = load_dataset(args.data, args.manifest, "evaluate")
    if set(dataset.smiles) & set(model.train_smiles):
        raise ValueError("External evaluation contains training molecules")
    strategy = model.provenance.get("split_strategy", "scaffold")
    if strategy == "scaffold":
        if {scaffold_key(s) for s in dataset.smiles} & {
            scaffold_key(s) for s in model.train_smiles
        }:
            raise ValueError("External scaffold evaluation overlaps training scaffolds")
    else:
        if any(not s for s in dataset.series):
            raise ValueError("External series evaluation requires all series IDs")
        if set(dataset.series) & set(model.provenance["training_series"]):
            raise ValueError("External evaluation overlaps training polymer families")
    import numpy as np

    from .splitting import grouping_keys

    groups = grouping_keys(dataset.smiles, dataset.series, strategy)
    report = evaluate(
        model, dataset, np.arange(len(dataset.smiles)), groups, configured_encoder(args)
    )
    report["evaluation_dataset_sha256"] = dataset.manifest.sha256
    if Path(args.report).exists():
        raise ValueError("Report output exists")
    write_json(args.report, report)


def auxiliary_training(args):
    from .auxiliary import AuxiliaryManifest, fit_auxiliary, save_auxiliary
    from .embargo import HoldoutEmbargo
    from .provenance import file_hash

    if Path(args.model).exists() or Path(args.report).exists():
        raise ValueError("Auxiliary outputs exist; choose fresh files")
    reserved = json.loads(Path(args.reserved_identities).read_text(encoding="utf-8"))
    if not isinstance(reserved, dict) or not isinstance(reserved.get("smiles"), list):
        raise ValueError("Reserved identities must be a JSON object with a smiles list")
    if not reserved["smiles"]:
        raise ValueError("Explicit nonempty HSP holdout identities are required")
    embargo = HoldoutEmbargo.from_smiles(reserved["smiles"], reserved.get("series", ()))
    manifest = AuxiliaryManifest.model_validate_json(
        Path(args.manifest).read_text(encoding="utf-8")
    )
    state, report = fit_auxiliary(args.data, manifest, configured_encoder(args), embargo, args.seed)
    report["reserved_identities_sha256"] = file_hash(args.reserved_identities)
    save_auxiliary(state, args.model)
    write_json(args.report, report)
    print(json.dumps(report))


def solquest_preparation(args):
    from .auxiliary import AuxiliaryManifest
    from .embargo import HoldoutEmbargo
    from .provenance import file_hash
    from .solquest import prepare_solquest

    reserved = json.loads(Path(args.reserved_identities).read_text(encoding="utf-8"))
    if not isinstance(reserved, dict) or not isinstance(reserved.get("smiles"), list):
        raise ValueError("Reserved identities must be a JSON object with a smiles list")
    if not reserved["smiles"]:
        raise ValueError("Explicit nonempty HSP holdout identities are required")
    embargo = HoldoutEmbargo.from_smiles(reserved["smiles"], reserved.get("series", ()))
    review = AuxiliaryManifest.model_validate_json(
        Path(args.source_review).read_text(encoding="utf-8")
    )
    report = prepare_solquest(args.source, args.out, review, embargo)
    report["reserved_identities_sha256"] = file_hash(args.reserved_identities)
    write_json(Path(args.out) / "preparation.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"source_row_indices", "rejected_source_rows"}
            }
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hansen research PoC. Synthetic metrics are not real accuracy."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser(
        "synthetic", help="Generate original synthetic structures/labels"
    )
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--seed", type=int, default=42)
    fetch = commands.add_parser(
        "checkpoint-fetch", help="Explicitly fetch the audited encoder bytes"
    )
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--checkpoint-review", required=True)
    solquest = commands.add_parser(
        "solquest-prepare", help="Prepare reviewed local computed-solvation targets without fitting"
    )
    solquest.add_argument(
        "--source", required=True, help="Local flat SolQuest JSON or one-JSON ZIP"
    )
    solquest.add_argument(
        "--source-review", required=True, help="Raw-source AuxiliaryManifest JSON"
    )
    solquest.add_argument("--reserved-identities", required=True, help="Full HSP/Excel smiles JSON")
    solquest.add_argument("--out", required=True)
    auxiliary = commands.add_parser(
        "auxiliary-fit", help="Train quantum/reference auxiliary targets with an HSP embargo"
    )
    auxiliary.add_argument("--data", required=True, help="NPZ: smiles, targets, target_names")
    auxiliary.add_argument("--manifest", required=True)
    auxiliary.add_argument(
        "--reserved-identities", required=True, help="JSON: smiles, optional series"
    )
    auxiliary.add_argument("--model", required=True)
    auxiliary.add_argument("--report", required=True)
    auxiliary.add_argument("--seed", type=int, default=42)
    add_encoder(auxiliary)
    features = commands.add_parser(
        "features", help="Normalize and inspect atom ownership and features"
    )
    features.add_argument("--smiles", required=True)
    schema = commands.add_parser(
        "schema", help="Write molecule/polymer/surfactant/mixture JSON Schema"
    )
    schema.add_argument("--output", required=True)
    validate = commands.add_parser(
        "validate", help="Validate input JSON and assess scope without predicting"
    )
    validate.add_argument("--input", required=True)
    chain = commands.add_parser(
        "chain-features", help="Assemble bounded chain representatives and audit intensive features"
    )
    chain.add_argument("--input", required=True)
    chain.add_argument("--output", required=True)
    reference = commands.add_parser(
        "reference-gc", help="Opt-in local coefficient reference; not validated HSP inference"
    )
    reference.add_argument("--parameters", required=True)
    reference.add_argument("--review", required=True)
    structure = reference.add_mutually_exclusive_group(required=True)
    structure.add_argument("--smiles")
    structure.add_argument("--repeat-unit")
    reference.add_argument("--repeat-count", type=int)
    reference.add_argument("--left-cap", help="Port SMILES or explicit 'hydrogen'")
    reference.add_argument("--right-cap", help="Port SMILES or explicit 'hydrogen'")
    cohesion = commands.add_parser(
        "md-cohesion-prepare", help="Prepare local MD observables; not three-component HSP labels"
    )
    cohesion.add_argument("--data", required=True)
    cohesion.add_argument("--review", required=True)
    cohesion.add_argument("--out", required=True)
    train = commands.add_parser("train")
    add_training(train)
    train.add_argument("--mode", choices=("A", "B", "C"), default="A")
    train.add_argument("--head", choices=("ridge", "lightgbm"), default="ridge")
    train.add_argument("--model", required=True)
    train.add_argument("--report", required=True)
    compare = commands.add_parser(
        "compare", help="Compare A, B-proxy, C-proxy on identical holdouts"
    )
    add_training(compare)
    compare.add_argument("--out", required=True)
    compare.add_argument("--include-lightgbm", action="store_true")
    predict = commands.add_parser("predict", help="Predict supported in-domain rows; refuse others")
    add_encoder(predict)
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    evaluate_parser = commands.add_parser(
        "evaluate", help="Evaluate independent local holdout; separate label-kind report"
    )
    add_encoder(evaluate_parser)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            paths = generate_synthetic(args.out, args.seed)
            print(
                json.dumps(
                    {"data": str(paths[0]), "manifest": str(paths[1]), "label_kind": "synthetic"}
                )
            )
        elif args.command == "checkpoint-fetch":
            from .molformer import fetch_checkpoint

            directory = fetch_checkpoint(args.out, args.checkpoint_review)
            print(json.dumps({"checkpoint": str(directory), "network_requested_explicitly": True}))
        elif args.command == "solquest-prepare":
            solquest_preparation(args)
        elif args.command == "auxiliary-fit":
            auxiliary_training(args)
        elif args.command == "features":
            coverage = atom_coverage(args.smiles)
            output = {**asdict(coverage), "coverage_fraction": coverage.fraction}
            output["features"] = (
                dict(zip(FEATURE_NAMES, chemical_features(args.smiles), strict=True))
                if not coverage.unassigned and coverage.fraction == 1
                else None
            )
            print(json.dumps(output))
        elif args.command == "schema":
            if Path(args.output).exists():
                raise ValueError("Schema output exists")
            write_json(args.output, INPUT_ADAPTER.json_schema())
        elif args.command == "validate":
            record = INPUT_ADAPTER.validate_json(Path(args.input).read_text(encoding="utf-8"))
            print(json.dumps({"kind": record.kind, **asdict(assess_scope(record))}))
        elif args.command == "chain-features":
            from .chains import characterize_homopolymer
            from .intensive import INTENSIVE_NAMES
            from .schema import PolymerInput, SurfactantInput
            from .surfactants import characterize_surfactant

            if Path(args.output).exists():
                raise ValueError("Chain report exists; choose a fresh output")
            record = INPUT_ADAPTER.validate_json(Path(args.input).read_text(encoding="utf-8"))
            if isinstance(record, PolymerInput):
                report = characterize_homopolymer(record)
            elif isinstance(record, SurfactantInput):
                report = characterize_surfactant(record)
            else:
                raise ValueError("chain-features requires a polymer or surfactant record")
            report["feature_names"] = INTENSIVE_NAMES
            write_json(args.output, report)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "hsp_predictions": None,
                        "physical_accuracy_validated": False,
                        "issues": report["issues"],
                    }
                )
            )
        elif args.command == "reference-gc":
            from .reference_gc import ReferenceGC

            calculator = ReferenceGC(args.parameters, args.review)
            if args.smiles is not None:
                if any(x is not None for x in (args.repeat_count, args.left_cap, args.right_cap)):
                    raise ValueError("Molecule calculation cannot include repeat/cap options")
                report = calculator.molecule(args.smiles)
            else:
                if any(x is None for x in (args.repeat_count, args.left_cap, args.right_cap)):
                    raise ValueError("Repeat calculation requires count and both explicit caps")
                report = calculator.homopolymer(
                    args.repeat_unit,
                    args.repeat_count,
                    left_cap=None if args.left_cap == "hydrogen" else args.left_cap,
                    right_cap=None if args.right_cap == "hydrogen" else args.right_cap,
                )
            print(json.dumps(report))
        elif args.command == "md-cohesion-prepare":
            from .cohesion import prepare_md_cohesion

            report = prepare_md_cohesion(args.data, args.review, args.out)
            print(
                json.dumps(
                    {
                        "out": args.out,
                        "counts": report["counts"],
                        "hsp_three_component_targets": 0,
                        "training_allowed": False,
                    }
                )
            )
        elif args.command in {"train", "compare"}:
            dataset = load_dataset(args.data, args.manifest)
            dataset.manifest.authorize("evaluate")
            split = split_dataset(dataset, args.split, args.seed)
            if args.command == "train":
                if Path(args.model).exists() or Path(args.report).exists():
                    raise ValueError("Output exists; choose fresh model and report files")
                model, report = fit_and_report(dataset, split, args, args.mode, args.head)
                model.save(args.model)
                write_json(args.report, report)
                print(
                    json.dumps(
                        {
                            "mode": args.mode,
                            "label_kind": dataset.manifest.label_kind,
                            "report": args.report,
                        }
                    )
                )
            else:
                directory = Path(args.out)
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError("Comparison directory must be new or empty")
                directory.mkdir(parents=True, exist_ok=True)
                reports = {}
                encoder = configured_encoder(args)
                choices = [("A", "ridge"), ("B", "ridge"), ("C", "ridge")]
                if args.include_lightgbm:
                    choices.append(("A", "lightgbm"))
                for mode, head in choices:
                    model, report = fit_and_report(dataset, split, args, mode, head, encoder)
                    name = f"{mode}-{head}"
                    model.save(directory / f"{name}.json")
                    reports[name] = report
                write_json(directory / "comparison.json", reports)
                print(
                    json.dumps(
                        {
                            "models": list(reports),
                            "label_kind": dataset.manifest.label_kind,
                            "report": str(directory / "comparison.json"),
                        }
                    )
                )
        elif args.command == "predict":
            predict_csv(args)
        elif args.command == "evaluate":
            external_evaluation(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"hansenkit: {exc}\n")


if __name__ == "__main__":
    main()
