"""Evaluation runner.

Usage::

    python -m evaluation.run --entity-type Occupation --files-dir path/to/inference/files
    python -m evaluation.run --entity-type Skill --files-dir path/to/inference/files
    python -m evaluation.run --entity-type Qualification --files-dir path/to/inference/files
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Run NEL evaluation")
    parser.add_argument(
        "--entity-type",
        choices=["Occupation", "Skill", "Qualification"],
        required=True,
        help="Entity type to evaluate",
    )
    parser.add_argument(
        "--files-dir",
        required=True,
        help="Path to inference/files directory (containing CSVs and embeddings)",
    )
    parser.add_argument("--output-json", help="Write metrics to this JSON file")
    args = parser.parse_args()

    from evaluation.nel_evaluator import NELEvaluator

    print(f"Running {args.entity_type} evaluation against {args.files_dir} ...")
    evaluator = NELEvaluator(entity_type=args.entity_type, files_dir=args.files_dir)
    metrics = evaluator.output

    print(json.dumps(metrics, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {args.output_json}")


if __name__ == "__main__":
    main()
