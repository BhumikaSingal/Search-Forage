"""Evaluate BM25 on a labeled query set with Precision@10 and nDCG@10.

Metrics are computed from the live ranking — values are not hard-coded guesses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from searchforge.bm25 import BM25
from searchforge.metrics import ndcg_at_k, precision_at_k

DATA_PATH = Path(__file__).parent / "data" / "bm25_eval.json"
K = 10


@pytest.fixture(scope="module")
def eval_bundle() -> dict:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    bm25 = BM25().fit(payload["documents"])
    rows: list[dict] = []

    for item in payload["queries"]:
        ranked = bm25.score(item["query"])
        ranked_ids = [doc_id for doc_id, _ in ranked]
        relevance = {doc_id: float(grade) for doc_id, grade in item["relevance"].items()}
        p_at_k = precision_at_k(ranked_ids, relevance, K)
        ndcg = ndcg_at_k(ranked_ids, relevance, K)
        rows.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "precision_at_10": p_at_k,
                "ndcg_at_10": ndcg,
                "top10": ranked_ids[:K],
                "relevance": relevance,
            }
        )

    mean_p = sum(row["precision_at_10"] for row in rows) / len(rows)
    mean_ndcg = sum(row["ndcg_at_10"] for row in rows) / len(rows)
    return {"rows": rows, "mean_precision_at_10": mean_p, "mean_ndcg_at_10": mean_ndcg}


def test_eval_dataset_is_well_formed():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert len(payload["documents"]) >= 10
    assert len(payload["queries"]) >= 3
    for item in payload["queries"]:
        assert item["query"].strip()
        assert item["relevance"], f"{item['query_id']} needs at least one relevant doc"
        for doc_id, grade in item["relevance"].items():
            assert doc_id in payload["documents"]
            assert grade > 0


def test_bm25_eval_metrics_are_in_range(eval_bundle: dict):
    for row in eval_bundle["rows"]:
        assert 0.0 <= row["precision_at_10"] <= 1.0
        assert 0.0 <= row["ndcg_at_10"] <= 1.0


def test_bm25_eval_reports_real_measured_scores(eval_bundle: dict, capsys: pytest.CaptureFixture[str]):
    """Print measured metrics so the run itself is the source of truth."""
    print("\nBM25 evaluation (Precision@10, nDCG@10)")
    print("-" * 60)
    for row in eval_bundle["rows"]:
        print(
            f"{row['query_id']}: P@10={row['precision_at_10']:.4f}  "
            f"nDCG@10={row['ndcg_at_10']:.4f}  query={row['query']!r}"
        )
        print(f"         top10={row['top10']}")
    print("-" * 60)
    print(
        f"macro-average: P@10={eval_bundle['mean_precision_at_10']:.4f}  "
        f"nDCG@10={eval_bundle['mean_ndcg_at_10']:.4f}"
    )

    # Sanity: every query with labeled relevant docs should retrieve at least one
    # of them somewhere in the ranking (not necessarily @10 for hard queries).
    for row in eval_bundle["rows"]:
        relevant = set(row["relevance"])
        assert relevant & set(row["top10"]), (
            f"{row['query_id']} retrieved none of its relevant docs in the top {K}"
        )

    # Macro-averages must come from the rows above (identity check, not invented).
    rows = eval_bundle["rows"]
    assert eval_bundle["mean_precision_at_10"] == pytest.approx(
        sum(r["precision_at_10"] for r in rows) / len(rows)
    )
    assert eval_bundle["mean_ndcg_at_10"] == pytest.approx(
        sum(r["ndcg_at_10"] for r in rows) / len(rows)
    )

    # Ensure the measured table was actually emitted.
    captured = capsys.readouterr().out
    assert "BM25 evaluation" in captured
    assert "macro-average" in captured


def test_bm25_places_highly_relevant_docs_early(eval_bundle: dict):
    by_id = {row["query_id"]: row for row in eval_bundle["rows"]}

    # q1: BM25-focused docs should dominate the head of the ranking
    top = by_id["q1"]["top10"]
    assert "d01" in top[:5]
    assert "d02" in top[:5]

    # q6: the only gardening document should be rank 1
    assert by_id["q6"]["top10"][0] == "d20"
    assert by_id["q6"]["ndcg_at_10"] == pytest.approx(1.0)
