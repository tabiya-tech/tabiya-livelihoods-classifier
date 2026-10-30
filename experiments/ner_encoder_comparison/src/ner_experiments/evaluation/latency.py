"""How fast a trained model runs on CPU, one ad at a time (as in production)."""

import statistics
import time
from collections import defaultdict
from pathlib import Path

import psutil
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification

from ner_experiments.modeling.prediction import predict_sentences
from ner_experiments.modeling.tokenization import load_tokenizer
from ner_experiments.schemas import LatencyResult, TaggedSentence


def benchmark_cpu_latency(
    model_dir: Path,
    test_sentences: list[TaggedSentence],
    threads: int = 4,
    warmup_ads: int = 10,
    max_pieces: int = 256,
) -> LatencyResult:
    torch.set_num_threads(threads)
    tokenizer = load_tokenizer(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to("cpu")
    model.eval()

    sentences_by_ad: dict[str, list[TaggedSentence]] = defaultdict(list)
    for sentence in test_sentences:
        sentences_by_ad[sentence.ad_id].append(sentence)
    ads = [sentences_by_ad[ad_id] for ad_id in sorted(sentences_by_ad)]

    for ad_sentences in ads[:warmup_ads]:
        predict_sentences(model, tokenizer, ad_sentences, max_pieces, "cpu", description="Warm-up")

    milliseconds_per_ad: list[float] = []
    for ad_sentences in tqdm(ads, desc="Timing ads on CPU", unit="ad"):
        start_time = time.perf_counter()
        predict_sentences(model, tokenizer, ad_sentences, max_pieces, "cpu", description="ad")
        milliseconds_per_ad.append((time.perf_counter() - start_time) * 1000)

    return LatencyResult(
        model_dir=str(model_dir),
        threads=threads,
        ads_timed=len(milliseconds_per_ad),
        median_ms_per_ad=statistics.median(milliseconds_per_ad),
        p95_ms_per_ad=statistics.quantiles(milliseconds_per_ad, n=20)[-1],
        peak_memory_mb=psutil.Process().memory_info().rss / 1024**2,
        parameters_millions=sum(parameter.numel() for parameter in model.parameters()) / 1e6,
    )
