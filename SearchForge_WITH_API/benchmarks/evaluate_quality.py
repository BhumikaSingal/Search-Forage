"""Evaluate BM25 and a simple term-overlap baseline on judged queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from searchforge.bm25 import BM25
from searchforge.metrics import ndcg_at_k, precision_at_k
from searchforge.tokenizer import tokenize

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "tests" / "data" / "bm25_eval.json"
K = 10


def term_overlap_ranking(
    documents: dict[str, str],
    query: str,
) -> list[tuple[str, int]]:
    """Rank OR-matching documents by unique query-token overlap, then ID."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    scores = []
    for doc_id, text in documents.items():
        overlap = len(query_terms.intersection(tokenize(text)))
        if overlap:
            scores.append((doc_id, overlap))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def evaluate_dataset(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    """Return per-query and macro P@10/nDCG@10 from a judged JSON dataset."""
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents: dict[str, str] = payload["documents"]
    ranker = BM25().fit(documents)
    rows: list[dict[str, Any]] = []

    for item in payload["queries"]:
        query = item["query"]
        relevance = {
            doc_id: float(grade)
            for doc_id, grade in item["relevance"].items()
        }
        query_terms = set(tokenize(query))
        candidates = {
            doc_id
            for doc_id, text in documents.items()
            if query_terms.intersection(tokenize(text))
        }
        bm25_ids = [
            doc_id
            for doc_id, _ in ranker.score_documents(query, candidates)
        ]
        overlap_ids = [
            doc_id for doc_id, _ in term_overlap_ranking(documents, query)
        ]
        rows.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "relevant_documents": len(relevance),
                "bm25": {
                    "precision_at_10": precision_at_k(bm25_ids, relevance, K),
                    "ndcg_at_10": ndcg_at_k(bm25_ids, relevance, K),
                    "top_10": bm25_ids[:K],
                },
                "term_overlap": {
                    "precision_at_10": precision_at_k(overlap_ids, relevance, K),
                    "ndcg_at_10": ndcg_at_k(overlap_ids, relevance, K),
                    "top_10": overlap_ids[:K],
                },
            }
        )

    return {
        "dataset": str(dataset_path),
        "document_count": len(documents),
        "query_count": len(rows),
        "k": K,
        "methodology": {
            "candidate_matching": "Both methods rank the same documents containing at least one normalized query token (OR semantics).",
            "bm25": "SearchForge Okapi BM25 with corpus-wide statistics.",
            "term_overlap": "Count of distinct normalized query tokens present in a document; ties are ordered by ascending document ID.",
            "metrics": "Precision@10 uses a fixed denominator of 10; nDCG@10 uses graded gain. Queries with no relevant judgments score zero.",
            "limitations": "Small hand-judged collection; results are illustrative and do not establish general search quality.",
        },
        "queries": rows,
        "macro_average": {
            method: {
                metric: mean(float(row[method][metric]) for row in rows)
                for metric in ("precision_at_10", "ndcg_at_10")
            }
            for method in ("bm25", "term_overlap")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.dumps(evaluate_dataset(args.dataset), indent=2) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote quality results to {args.output}")


if __name__ == "__main__":
    main()