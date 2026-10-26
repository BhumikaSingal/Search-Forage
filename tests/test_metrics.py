"""Unit tests for ranking metrics with hand-checked examples."""

import math

import pytest

from searchforge.metrics import dcg_at_k, ndcg_at_k, precision_at_k


def test_precision_at_k_hand_calculated():
    relevance = {"a": 1, "b": 0, "c": 2, "d": 1}
    ranked = ["a", "b", "c", "d", "e"]

    # top-3: a (rel), b (not), c (rel) → 2/3
    assert precision_at_k(ranked, relevance, 3) == pytest.approx(2 / 3)
    # top-1: a relevant → 1/1
    assert precision_at_k(ranked, relevance, 1) == pytest.approx(1.0)
    # top-10 pads with misses beyond ranked list length → 3 relevant / 10
    assert precision_at_k(ranked, relevance, 10) == pytest.approx(3 / 10)


def test_precision_at_k_edge_cases():
    assert precision_at_k([], {"a": 1}, 10) == 0.0
    assert precision_at_k(["a"], {"a": 1}, 0) == 0.0
    assert precision_at_k(["a", "b"], {}, 2) == 0.0


def test_ndcg_at_k_hand_calculated():
    relevance = {"d1": 3, "d2": 2, "d3": 0, "d4": 1}
    ranked = ["d2", "d1", "d3", "d4"]

    # DCG@3 = (2^2-1)/log2(2) + (2^3-1)/log2(3) + (2^0-1)/log2(4)
    dcg = (3.0 / math.log2(2)) + (7.0 / math.log2(3)) + (0.0 / math.log2(4))
    assert dcg_at_k(ranked, relevance, 3) == pytest.approx(dcg)

    # Ideal order by grade: d1 (3), d2 (2), d4 (1)
    idcg = (7.0 / math.log2(2)) + (3.0 / math.log2(3)) + (1.0 / math.log2(4))
    assert ndcg_at_k(ranked, relevance, 3) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], relevance, 2) == pytest.approx(1.0)


def test_ndcg_no_relevant_documents_is_zero():
    assert ndcg_at_k(["a", "b"], {"a": 0, "c": 0}, 10) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0
