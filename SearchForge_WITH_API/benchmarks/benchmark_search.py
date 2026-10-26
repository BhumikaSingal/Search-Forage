"""Reproducible synthetic benchmark for the public SearchForge search API."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from searchforge.bm25 import BM25
from searchforge.metrics import ndcg_at_k, precision_at_k
from searchforge.search import SearchEngine
from searchforge.tokenizer import tokenize

DEFAULT_SEED = 20261026
DEFAULT_SIZES = (1_000, 5_000, 10_000)
DEFAULT_RUNS = 100
DEFAULT_WARMUPS = 5
K = 10
COMMON_TERMS = ("search", "document", "common")
FILLER_TERMS = (
    "alpha", "archive", "article", "author", "catalog", "chapter", "collection",
    "content", "data", "digital", "field", "guide", "history", "information",
    "language", "library", "method", "paper", "record", "reference", "report",
    "research", "resource", "science", "system", "text", "topic", "volume",
)


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    relevance: Mapping[str, int]


QUERY_CASES = (
    QueryCase("q_bm25", "bm25 ranking", {"doc-00000": 3, "doc-00001": 1}),
    QueryCase("q_index", "inverted index", {"doc-00002": 3}),
    QueryCase("q_tokenizer", "token normalization", {"doc-00003": 2}),
    QueryCase("q_rare", "quasar", {"doc-00004": 3}),
    QueryCase("q_common_unjudged", "common document", {}),
    QueryCase("q_unknown", "unseenword", {}),
)

PLANTED_TEXT = {
    "doc-00000": "bm25 ranking retrieval algorithm",
    "doc-00001": "bm25 ranking relevance scoring",
    "doc-00002": "inverted index postings retrieval",
    "doc-00003": "token normalization unicode text",
    "doc-00004": "quasar rareterm astronomy",
}


def generate_corpus(size: int, seed: int = DEFAULT_SEED) -> dict[str, str]:
    """Generate a fixed-seed corpus with varied filler and planted topic docs."""
    if size < len(PLANTED_TEXT):
        raise ValueError(f"size must be at least {len(PLANTED_TEXT)}")

    rng = random.Random(seed)
    documents: dict[str, str] = {}
    for index in range(size):
        doc_id = f"doc-{index:05d}"
        filler_count = 5 + ((index * 17 + rng.randrange(41)) % 41)
        filler = " ".join(rng.choice(FILLER_TERMS) for _ in range(filler_count))
        topic = PLANTED_TEXT.get(doc_id, "")
        documents[doc_id] = " ".join(part for part in (" ".join(COMMON_TERMS), topic, filler) if part)
    return documents


def _percentile(samples: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _processor_name() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() == "model name":
                    return value.strip()
        except OSError:
            pass

    return platform.processor() or platform.uname().processor or platform.machine()


def _measure_query(
    search: Callable[[str], list[tuple[str, float]]],
    query_case: QueryCase,
    runs: int,
    warmups: int,
) -> dict[str, object]:
    for _ in range(warmups):
        search(query_case.query)

    elapsed_ms: list[float] = []
    ranked_ids: list[str] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        ranked = search(query_case.query)
        elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        ranked_ids = [doc_id for doc_id, _ in ranked[:K]]

    mean_ms = statistics.mean(elapsed_ms)

    return {
        "query_id": query_case.query_id,
        "query": query_case.query,
        "relevant_documents": len(query_case.relevance),
        "precision_at_10": precision_at_k(ranked_ids, query_case.relevance, K),
        "ndcg_at_10": ndcg_at_k(ranked_ids, query_case.relevance, K),
        "latency_ms": {
            "p50": _percentile(elapsed_ms, 0.50),
            "p95": _percentile(elapsed_ms, 0.95),
            "runs": runs,
            "throughput_queries_per_second": 1_000 / mean_ms,
        },
        "top_10": ranked_ids,
    }


def full_scan_search(
    documents: Mapping[str, str],
    ranker: BM25,
    query: str,
) -> list[tuple[str, float]]:
    """Scan all documents for OR-matching tokens, then use the same BM25 ranker."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    candidates = {
        doc_id
        for doc_id, text in documents.items()
        if query_terms.intersection(tokenize(text))
    }
    if not candidates:
        return []
    return ranker.score_documents(query, candidates)


def run_benchmark(
    sizes: Sequence[int] = DEFAULT_SIZES,
    *,
    seed: int = DEFAULT_SEED,
    runs: int = DEFAULT_RUNS,
    warmups: int = DEFAULT_WARMUPS,
) -> dict[str, object]:
    """Run quality and warmed search-latency measurements at each size."""
    if not sizes or any(size < len(PLANTED_TEXT) for size in sizes):
        raise ValueError(f"sizes must be non-empty and at least {len(PLANTED_TEXT)}")
    if runs < 1 or warmups < 0:
        raise ValueError("runs must be positive and warmups must be non-negative")

    size_results = []
    for size in sizes:
        documents = generate_corpus(size, seed)
        engine = SearchEngine()
        build_start = time.perf_counter()
        engine.add_documents(documents)
        indexed_build_seconds = time.perf_counter() - build_start

        full_scan_ranker = BM25()
        build_start = time.perf_counter()
        full_scan_ranker.fit(documents)
        full_scan_build_seconds = time.perf_counter() - build_start

        def indexed_search(query: str) -> list[tuple[str, float]]:
            return engine.search(query)

        def scan_search(query: str) -> list[tuple[str, float]]:
            return full_scan_search(documents, full_scan_ranker, query)

        query_results = []
        for query_case in QUERY_CASES:
            indexed_result = indexed_search(query_case.query)
            scan_result = scan_search(query_case.query)
            if indexed_result != scan_result:
                raise RuntimeError(
                    f"Full-scan and indexed rankings differ for {query_case.query!r}"
                )

            indexed_metrics = _measure_query(
                indexed_search, query_case, runs, warmups
            )
            scan_metrics = _measure_query(scan_search, query_case, runs, warmups)
            query_results.append(
                {
                    **indexed_metrics,
                    "methods": {
                        "inverted_index": indexed_metrics,
                        "full_scan": scan_metrics,
                    },
                }
            )

        size_results.append(
            {
                "document_count": engine.document_count,
                "vocabulary_size": engine.vocabulary_size,
                "build_seconds": {
                    "inverted_index_and_bm25": indexed_build_seconds,
                    "full_scan_bm25_only": full_scan_build_seconds,
                },
                "index_build_seconds": indexed_build_seconds,
                "queries": query_results,
                "macro_average": {
                    "precision_at_10": statistics.mean(
                        float(row["precision_at_10"]) for row in query_results
                    ),
                    "ndcg_at_10": statistics.mean(
                        float(row["ndcg_at_10"]) for row in query_results
                    ),
                },
            }
        )

    return {
        "benchmark": "searchforge_synthetic_search_v1",
        "seed": seed,
        "k": K,
        "warmup_searches_per_query": warmups,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": _processor_name(),
            "logical_cpu_count": os.cpu_count(),
        },
        "methodology": {
            "corpus": "Seeded synthetic documents with three shared terms, 5-45 filler terms, and five fixed topic documents.",
            "matching": "Both methods return documents containing at least one normalized query token (OR semantics), use identical corpus-wide BM25 scoring, and tie-break by document ID. A parity check runs before timing each query.",
            "full_scan": "Tokenizes and checks every document per query, then scores matching documents with the same BM25 implementation.",
            "latency": "Build and warm-up are excluded. Inverted-index measurements use SearchEngine.search; full-scan measurements include scanning, tokenization, BM25 scoring, and sorting. Timed calls use perf_counter_ns.",
            "percentiles": "Nearest-rank p50 and p95 over repeated query calls.",
            "throughput": "Measured query count divided by total timed duration for each query/method; sequential single-process throughput, not concurrent capacity.",
            "build_time": "Inverted build includes postings and BM25; full-scan baseline build includes BM25 corpus-statistic fitting but no postings index.",
            "precision_at_10": "Missing result slots count as non-relevant; queries with no judgments score 0.",
            "ndcg_at_10": "Graded gain; queries with no positive judgments score 0.",
            "limitations": "Synthetic data and fixed judgments are for repeatable local comparisons, not a representation of real-world web search or production workloads.",
        },
        "sizes": size_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()

    results = run_benchmark(args.sizes, seed=args.seed, runs=args.runs, warmups=args.warmups)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()