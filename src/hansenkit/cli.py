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
    model = train_model(dataset, split.train, split.groups, mode, encoder, head, args.seed)
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
