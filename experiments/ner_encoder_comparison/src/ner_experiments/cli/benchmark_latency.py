"""CPU speed of a trained model (needs keep_model: true on that run).

    python -m ner_experiments.cli.benchmark_latency --model-dir runs/<run name>/model
"""

import argparse
from pathlib import Path

from ner_experiments.config import DataPaths
from ner_experiments.corpus.loading import read_sentences
from ner_experiments.evaluation.latency import benchmark_cpu_latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()

    test_sentences = read_sentences(DataPaths().processed_dir, "test")
    result = benchmark_cpu_latency(arguments.model_dir, test_sentences, threads=arguments.threads)
    print(result.model_dump_json(indent=2))
    output_path = arguments.model_dir.parent / "latency.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
