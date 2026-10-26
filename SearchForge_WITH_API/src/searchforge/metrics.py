"""Ranking evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def precision_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Binary Precision@k: fraction of the top-k results that are relevant.

    A document is relevant when its graded relevance is ``> 0``. If ``k <= 0``
    or there are no ranked results to consider, the score is ``0.0``.
    """
    if k <= 0:
        return 0.0

    top = list(ranked_ids)[:k]
    if not top:
        return 0.0

    hits = sum(1 for doc_id in top if relevance.get(doc_id, 0) > 0)
    return hits / k


def dcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Discounted Cumulative Gain at k using graded relevance gains."""
    if k <= 0:
        return 0.0

    total = 0.0
    for rank, doc_id in enumerate(list(ranked_ids)[:k], start=1):
        rel = float(relevance.get(doc_id, 0))
        if rel <= 0:
            continue
        gain = (2.0**rel) - 1.0
        total += gain / math.log2(rank + 1)
    return total


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Normalized DCG@k; ``0.0`` when there are no relevant documents."""
    if k <= 0:
        return 0.0

    ideal = sorted(relevance.values(), reverse=True)
    ideal = [rel for rel in ideal if rel > 0][:k]
    if not ideal:
        return 0.0

    idcg = 0.0
    for rank, rel in enumerate(ideal, start=1):
        gain = (2.0**float(rel)) - 1.0
        idcg += gain / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg_at_k(ranked_ids, relevance, k) / idcg
