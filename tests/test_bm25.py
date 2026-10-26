"""Unit tests for Okapi BM25, including hand-calculated and reference checks."""

from __future__ import annotations

import math

import pytest
from rank_bm25 import BM25Okapi

from searchforge.bm25 import BM25
from searchforge.tokenizer import tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Five-document corpus chosen so query terms have positive raw IDF
# (appear in at most half the documents), keeping hand math simple.
HAND_DOCS = {
    "d1": "the cat sat",
    "d2": "the dog ran",
    "d3": "a bird flew",
    "d4": "the cat ran",
    "d5": "fish swim",
}


@pytest.fixture
def hand_bm25() -> BM25:
    return BM25(k1=1.5, b=0.75, epsilon=0.25).fit(HAND_DOCS)


def _okapi_idf(n_docs: int, df: int) -> float:
    """Raw ATIRE Okapi IDF without epsilon floor."""
    return math.log(n_docs - df + 0.5) - math.log(df + 0.5)


def _tf_component(tf: int, doc_len: int, avgdl: float, k1: float, b: float) -> float:
    length_norm = 1.0 - b + b * (doc_len / avgdl)
    return (tf * (k1 + 1.0)) / (tf + k1 * length_norm)


# ---------------------------------------------------------------------------
# Collection statistics
# ---------------------------------------------------------------------------


def test_empty_collection_stats():
    bm25 = BM25()
    assert bm25.document_count == 0
    assert bm25.average_document_length == 0.0
    assert bm25.document_frequency("cat") == 0
    assert bm25.score("cat") == []


def test_document_frequency_and_average_length(hand_bm25: BM25):
    # Tokens: 3+3+3+3+2 = 14; N = 5; avgdl = 2.8
    assert hand_bm25.document_count == 5
    assert hand_bm25.average_document_length == pytest.approx(2.8)
    assert hand_bm25.document_frequency("cat") == 2
    assert hand_bm25.document_frequency("the") == 3
    assert hand_bm25.document_frequency("fish") == 1
    assert hand_bm25.document_frequency("zebra") == 0
    assert hand_bm25.document_frequency("cat dog") == 0
    assert hand_bm25.document_frequency("") == 0


def test_term_frequency(hand_bm25: BM25):
    assert hand_bm25.term_frequency("d1", "cat") == 1
    assert hand_bm25.term_frequency("d4", "cat") == 1
    assert hand_bm25.term_frequency("d2", "cat") == 0
    assert hand_bm25.term_frequency("missing", "cat") == 0
    assert hand_bm25.term_frequency("d1", "unknown") == 0

    multi = BM25().fit({"x": "cat cat dog"})
    assert multi.term_frequency("x", "cat") == 2
    assert multi.term_frequency("x", "dog") == 1


def test_document_length(hand_bm25: BM25):
    assert hand_bm25.document_length("d1") == 3
    assert hand_bm25.document_length("d5") == 2
    assert hand_bm25.document_length("missing") is None


# ---------------------------------------------------------------------------
# Empty queries and unknown terms
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_ranking(hand_bm25: BM25):
    assert hand_bm25.score("") == []
    assert hand_bm25.score("   ") == []
    assert hand_bm25.score("...!!!") == []
    assert hand_bm25.score_document("", "d1") == 0.0


def test_unknown_terms_do_not_contribute(hand_bm25: BM25):
    # Pure unknown vocabulary → all scores stay 0; stable tie-break by doc_id
    ranked = hand_bm25.score("zebra unicorn")
    assert len(ranked) == 5
    assert all(score == 0.0 for _, score in ranked)
    assert [doc_id for doc_id, _ in ranked] == ["d1", "d2", "d3", "d4", "d5"]

    # Mixing known + unknown equals scoring the known term alone
    mixed = hand_bm25.score("cat zebra")
    known_only = hand_bm25.score("cat")
    assert [s for _, s in mixed] == pytest.approx([s for _, s in known_only])


# ---------------------------------------------------------------------------
# Hand-calculated BM25 scores
# ---------------------------------------------------------------------------


def test_hand_calculated_single_term_scores(hand_bm25: BM25):
    """Verify BM25('cat') against arithmetic from the Okapi formula."""
    n_docs = 5
    avgdl = 2.8
    k1, b = 1.5, 0.75
    df_cat = 2
    idf_cat = _okapi_idf(n_docs, df_cat)
    assert idf_cat > 0
    assert hand_bm25.idf("cat") == pytest.approx(idf_cat)

    expected = {
        "d1": idf_cat * _tf_component(1, 3, avgdl, k1, b),
        "d2": 0.0,
        "d3": 0.0,
        "d4": idf_cat * _tf_component(1, 3, avgdl, k1, b),
        "d5": 0.0,
    }

    ranked = hand_bm25.score("cat")
    got = dict(ranked)
    for doc_id, score in expected.items():
        assert got[doc_id] == pytest.approx(score)

    # d1 and d4 tie on score; tie-break is ascending doc_id
    assert [doc_id for doc_id, _ in ranked[:2]] == ["d1", "d4"]
    assert ranked[0][1] == pytest.approx(ranked[1][1])
    assert ranked[0][1] > 0


def test_hand_calculated_multi_term_with_length_effect():
    """Higher TF and shorter docs should outrank a single occurrence in a longer doc."""
    docs = {
        "short": "cat cat",           # len=2, tf(cat)=2
        "long": "cat dog bird fish",  # len=4, tf(cat)=1
        "none": "dog bird",           # tf(cat)=0
        "other1": "fish swim",
        "other2": "bird flew",
    }
    bm25 = BM25(k1=1.5, b=0.75).fit(docs)
    n_docs = 5
    avgdl = (2 + 4 + 2 + 2 + 2) / 5
    idf_cat = _okapi_idf(n_docs, df=2)
    # df=2 of N=5 → positive raw IDF (no epsilon floor)
    assert idf_cat > 0
    assert bm25.idf("cat") == pytest.approx(idf_cat)

    expected_short = idf_cat * _tf_component(2, 2, avgdl, 1.5, 0.75)
    expected_long = idf_cat * _tf_component(1, 4, avgdl, 1.5, 0.75)
    assert expected_short > expected_long

    ranked = bm25.score("cat")
    got = dict(ranked)
    assert got["short"] == pytest.approx(expected_short)
    assert got["long"] == pytest.approx(expected_long)
    assert got["none"] == 0.0
    assert ranked[0][0] == "short"


def test_hand_calculated_negative_idf_uses_epsilon_floor():
    """Terms in more than half the docs get the epsilon * average_idf floor."""
    docs = {
        "a": "the cat",
        "b": "the dog",
        "c": "the bird",
    }
    epsilon = 0.25
    bm25 = BM25(k1=1.5, b=0.75, epsilon=epsilon).fit(docs)

    # Raw IDF for "the" (df=3, N=3) is negative
    raw_the = _okapi_idf(3, 3)
    assert raw_the < 0

    # average_idf over {the, cat, dog, bird}
    raw_values = [
        _okapi_idf(3, 3),  # the
        _okapi_idf(3, 1),  # cat
        _okapi_idf(3, 1),  # dog
        _okapi_idf(3, 1),  # bird
    ]
    average_idf = sum(raw_values) / len(raw_values)
    expected_floor = epsilon * average_idf

    assert bm25.idf("the") == pytest.approx(expected_floor)
    # Floor equals epsilon * average_idf (may be negative when average_idf is).
    assert bm25.idf("the") != pytest.approx(raw_the)


# ---------------------------------------------------------------------------
# Trusted reference: rank_bm25.BM25Okapi
# ---------------------------------------------------------------------------


def test_scores_match_rank_bm25_okapi_reference(hand_bm25: BM25):
    doc_ids = sorted(HAND_DOCS)
    corpus = [tokenize(HAND_DOCS[doc_id]) for doc_id in doc_ids]
    reference = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)

    for query in ("cat", "the cat", "fish swim", "zebra", "cat dog ran"):
        query_tokens = tokenize(query)
        ref_scores = reference.get_scores(query_tokens)
        ours = dict(hand_bm25.score(query))

        for i, doc_id in enumerate(doc_ids):
            assert ours[doc_id] == pytest.approx(float(ref_scores[i]), rel=1e-12, abs=1e-12)


def test_scores_match_rank_bm25_on_eval_shaped_corpus():
    docs = {
        "d1": "BM25 ranking with term frequency and document length.",
        "d2": "Boolean inverted index without ranking scores.",
        "d3": "Precision and nDCG evaluate ranked retrieval quality.",
        "d4": "Python FastAPI service exposes a search HTTP API.",
    }
    bm25 = BM25().fit(docs)
    doc_ids = list(docs)
    reference = BM25Okapi(
        [tokenize(docs[doc_id]) for doc_id in doc_ids],
        k1=1.5,
        b=0.75,
        epsilon=0.25,
    )

    for query in (
        "BM25 term frequency",
        "precision ndcg",
        "python fastapi",
        "unknownxyz",
        "",
    ):
        query_tokens = tokenize(query)
        if not query_tokens:
            assert bm25.score(query) == []
            continue

        ref_scores = reference.get_scores(query_tokens)
        ours = dict(bm25.score(query))
        for i, doc_id in enumerate(doc_ids):
            assert ours[doc_id] == pytest.approx(float(ref_scores[i]), rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# API / validation
# ---------------------------------------------------------------------------


def test_fit_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        BM25().fit({"": "text"})
    with pytest.raises(TypeError):
        BM25().fit({"d": 123})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        BM25(k1=-1)
    with pytest.raises(ValueError):
        BM25(b=1.5)


@pytest.mark.parametrize("parameter", ["k1", "b", "epsilon"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_constructor_rejects_non_finite_parameters(parameter, value):
    with pytest.raises(ValueError, match="finite"):
        BM25(**{parameter: value})


def test_failed_fit_preserves_previous_corpus_and_scores():
    ranker = BM25().fit({"stable": "stable ranking terms"})
    previous_ranking = ranker.score("stable")
    previous_average_length = ranker.average_document_length

    with pytest.raises(TypeError, match="Document text must be a string"):
        ranker.fit({"valid": "partial replacement", "invalid": None})  # type: ignore[dict-item]

    assert ranker.document_ids == frozenset({"stable"})
    assert ranker.score("stable") == previous_ranking
    assert ranker.average_document_length == previous_average_length
    assert ranker.document_frequency("partial") == 0


def test_score_requires_string_query(hand_bm25: BM25):
    with pytest.raises(TypeError):
        hand_bm25.score(None)  # type: ignore[arg-type]
