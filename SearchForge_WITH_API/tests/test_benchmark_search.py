"""Tests for reproducible synthetic SearchForge benchmark inputs and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_search import (  # noqa: E402
    QUERY_CASES,
    full_scan_search,
    generate_corpus,
    run_benchmark,
)
from benchmarks.evaluate_quality import (  # noqa: E402
    evaluate_dataset,
    term_overlap_ranking,
)
from searchforge.bm25 import BM25  # noqa: E402
from searchforge.search import SearchEngine  # noqa: E402


def test_corpus_generation_is_deterministic_and_seeded():
    first = generate_corpus(20, seed=19)
    assert first == generate_corpus(20, seed=19)
    assert first != generate_corpus(20, seed=20)
    assert len({len(text.split()) for text in first.values()}) > 1
    assert all("search document common" in text for text in first.values())


def test_planted_relevance_documents_are_in_corpus():
    corpus = generate_corpus(20)
    for query_case in QUERY_CASES:
        assert set(query_case.relevance) <= corpus.keys()
    assert "unseenword" not in " ".join(corpus.values())


def test_benchmark_metrics_handle_short_and_unjudged_queries():
    result = run_benchmark([20], runs=1, warmups=0)
    queries = {row["query_id"]: row for row in result["sizes"][0]["queries"]}

    # A single relevant hit among fewer than ten results still uses P@10's fixed denominator.
    assert queries["q_rare"]["top_10"] == ["doc-00004"]
    assert queries["q_rare"]["precision_at_10"] == pytest.approx(0.1)
    assert queries["q_rare"]["ndcg_at_10"] == pytest.approx(1.0)
    assert queries["q_common_unjudged"]["precision_at_10"] == 0.0
    assert queries["q_common_unjudged"]["ndcg_at_10"] == 0.0
    assert queries["q_unknown"]["top_10"] == []
    assert set(queries["q_rare"]["methods"]) == {"inverted_index", "full_scan"}
    for method in queries["q_rare"]["methods"].values():
        assert method["latency_ms"]["p50"] >= 0
        assert method["latency_ms"]["p95"] >= method["latency_ms"]["p50"]
        assert method["latency_ms"]["throughput_queries_per_second"] > 0


def test_full_scan_matches_indexed_ranking_for_equivalent_candidates():
    documents = {
        "d1": "cat dog",
        "d2": "dog cat",
        "d3": "bird",
        "d4": "cat bird",
    }
    engine = SearchEngine()
    engine.add_documents(documents)
    ranker = BM25().fit(documents)

    for query in ("cat", "cat bird", "missing", "!!!"):
        assert full_scan_search(documents, ranker, query) == engine.search(query)


def test_term_overlap_baseline_has_deterministic_ties_and_empty_behavior():
    documents = {"d2": "cat dog", "d1": "dog cat", "d3": "cat"}
    assert term_overlap_ranking(documents, "cat dog") == [
        ("d1", 2),
        ("d2", 2),
        ("d3", 1),
    ]
    assert term_overlap_ranking(documents, "unknown") == []
    assert term_overlap_ranking(documents, "!!!") == []


def test_quality_report_calculates_both_methods_on_judged_collection():
    report = evaluate_dataset()
    assert report["document_count"] == 20
    assert report["query_count"] == 6
    for method in ("bm25", "term_overlap"):
        for metric in ("precision_at_10", "ndcg_at_10"):
            assert 0 <= report["macro_average"][method][metric] <= 1


def test_benchmark_rejects_invalid_sizes_and_runs():
    with pytest.raises(ValueError):
        generate_corpus(4)
    with pytest.raises(ValueError):
        run_benchmark([5], runs=0)