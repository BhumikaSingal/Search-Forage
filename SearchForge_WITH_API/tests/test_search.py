"""End-to-end search retrieval: index candidates + corpus-wide BM25 ranking."""

from __future__ import annotations

import math

import pytest

from searchforge.bm25 import BM25
from searchforge.search import SearchEngine
from searchforge.tokenizer import tokenize


# Hand-verifiable corpus (same as BM25 unit tests).
HAND_DOCS = {
    "d1": "the cat sat",
    "d2": "the dog ran",
    "d3": "a bird flew",
    "d4": "the cat ran",
    "d5": "fish swim",
}


def _okapi_idf(n_docs: int, df: int) -> float:
    return math.log(n_docs - df + 0.5) - math.log(df + 0.5)


def _tf_component(tf: int, doc_len: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    length_norm = 1.0 - b + b * (doc_len / avgdl)
    return (tf * (k1 + 1.0)) / (tf + k1 * length_norm)


@pytest.fixture
def engine() -> SearchEngine:
    eng = SearchEngine(k1=1.5, b=0.75, epsilon=0.25)
    for doc_id, text in HAND_DOCS.items():
        eng.add_document(doc_id, text)
    return eng


# ---------------------------------------------------------------------------
# Hand-verifiable ranking
# ---------------------------------------------------------------------------


def test_search_ranking_matches_hand_calculated_scores(engine: SearchEngine):
    avgdl = 2.8
    idf_cat = _okapi_idf(5, 2)
    expected_hit = idf_cat * _tf_component(1, 3, avgdl)

    ranked = engine.search("cat")
    assert [doc_id for doc_id, _ in ranked] == ["d1", "d4"]
    assert ranked[0][1] == pytest.approx(expected_hit)
    assert ranked[1][1] == pytest.approx(expected_hit)
    # Only postings candidates — zero-score docs are not returned.
    assert {doc_id for doc_id, _ in ranked} == {"d1", "d4"}


def test_search_order_matches_full_bm25_on_candidates(engine: SearchEngine):
    query = "cat ran"
    full = BM25(k1=1.5, b=0.75, epsilon=0.25).fit(HAND_DOCS).score(query)
    positive = [(doc_id, score) for doc_id, score in full if score != 0.0]

    got = engine.search(query)
    assert [doc_id for doc_id, _ in got] == [doc_id for doc_id, _ in positive]
    for (_, got_score), (_, exp_score) in zip(got, positive, strict=True):
        assert got_score == pytest.approx(exp_score)


def test_search_uses_corpus_wide_stats_not_candidate_local(engine: SearchEngine):
    # Candidate set for "fish" is only d5, but DF/avgdl must still be corpus-wide.
    assert engine.document_frequency("fish") == 1
    assert engine.average_document_length == pytest.approx(2.8)
    assert engine.document_count == 5

    idf_fish = _okapi_idf(5, 1)
    expected = idf_fish * _tf_component(1, 2, 2.8)
    ranked = engine.search("fish")
    assert ranked == [("d5", pytest.approx(expected))]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_query_and_unknown_terms(engine: SearchEngine):
    assert engine.search("") == []
    assert engine.search("   ") == []
    assert engine.search("...!!!") == []
    assert engine.search("zebra unicorn") == []
    # Known + unknown → same as known alone
    assert [d for d, _ in engine.search("cat zebra")] == ["d1", "d4"]


def test_tokenless_documents_affect_stats_but_are_not_candidates():
    eng = SearchEngine()
    eng.add_document("empty", "")
    eng.add_document("punct", "???")
    eng.add_document("real", "cat sat")

    assert eng.document_count == 3
    # avgdl = (0 + 0 + 2) / 3
    assert eng.average_document_length == pytest.approx(2 / 3)
    assert eng.search("cat")[0][0] == "real"
    assert all(doc_id == "real" for doc_id, _ in eng.search("cat"))
    # Token-less docs never appear as hits.
    assert eng.search("???") == []


def test_deleted_documents_leave_no_stale_postings_or_scores(engine: SearchEngine):
    assert engine.delete_document("d1") is True
    assert engine.delete_document("d1") is False
    assert engine.get_document("d1") is None
    assert "d1" not in engine.document_ids

    ranked = engine.search("cat")
    assert [doc_id for doc_id, _ in ranked] == ["d4"]
    # Corpus stats refreshed: N=4, tokens=3+3+3+2=11, avgdl=2.75, df(cat)=1
    assert engine.document_count == 4
    assert engine.average_document_length == pytest.approx(11 / 4)
    assert engine.document_frequency("cat") == 1


def test_replace_document_updates_postings_and_ranking():
    eng = SearchEngine()
    eng.add_document("d1", "alpha beta")
    eng.add_document("d2", "beta gamma")
    assert [d for d, _ in eng.search("alpha")] == ["d1"]

    eng.add_document("d1", "gamma delta")  # replace
    assert eng.search("alpha") == []
    assert {d for d, _ in eng.search("gamma")} == {"d1", "d2"}
    assert eng.get_document("d1") == "gamma delta"


def test_add_documents_matches_incremental_ingestion():
    documents = {**HAND_DOCS, "d6": "cat fish"}
    batch = SearchEngine()
    batch.add_documents(documents)

    incremental = SearchEngine()
    for doc_id, text in documents.items():
        incremental.add_document(doc_id, text)

    assert batch.document_count == incremental.document_count
    assert batch.vocabulary_size == incremental.vocabulary_size
    for query in ("cat", "fish", "cat ran", "unknown"):
        assert batch.search(query) == incremental.search(query)


def test_add_documents_rejects_invalid_batch_without_partial_mutation():
    engine = SearchEngine()
    engine.add_document("existing", "stable indexed text")
    previous_ids = engine.document_ids

    with pytest.raises(TypeError, match="Document text must be a string"):
        engine.add_documents({"valid": "partial poison", "invalid": None})  # type: ignore[dict-item]

    assert engine.document_ids == previous_ids
    assert engine.search("partial poison") == []
    assert engine.document_frequency("partial") == 0
    assert engine.search("stable")


def test_search_limit_and_deterministic_tie_break(engine: SearchEngine):
    ranked = engine.search("cat", limit=1)
    assert ranked == [("d1", pytest.approx(ranked[0][1]))]

    # Equal scores → ascending doc_id
    full = engine.search("cat")
    assert full[0][0] == "d1"
    assert full[1][0] == "d4"
    assert full[0][1] == pytest.approx(full[1][1])


def test_search_rejects_invalid_limit(engine: SearchEngine):
    with pytest.raises(ValueError):
        engine.search("cat", limit=-1)
    with pytest.raises(TypeError):
        engine.search(None)  # type: ignore[arg-type]


def test_empty_engine_search_is_empty():
    eng = SearchEngine()
    assert eng.search("cat") == []
    assert eng.document_count == 0


# ---------------------------------------------------------------------------
# Integration: index updates keep BM25 aligned with a fresh fit
# ---------------------------------------------------------------------------


def test_incremental_updates_match_refit_bm25():
    eng = SearchEngine()
    docs = dict(HAND_DOCS)
    for doc_id, text in docs.items():
        eng.add_document(doc_id, text)

    eng.delete_document("d3")
    eng.add_document("d6", "cat fish")
    eng.add_document("d2", "the dog swam")  # replace

    live = {doc_id: eng.get_document(doc_id) for doc_id in eng.document_ids}
    assert None not in live.values()
    reference = BM25(k1=1.5, b=0.75, epsilon=0.25).fit(live)  # type: ignore[arg-type]

    for query in ("cat", "dog swam", "fish", "bird", "zebra"):
        got = eng.search(query)
        ref_full = reference.score(query)
        expected = [(doc_id, score) for doc_id, score in ref_full if score != 0.0]
        assert [doc_id for doc_id, _ in got] == [doc_id for doc_id, _ in expected]
        for (_, g), (_, e) in zip(got, expected, strict=True):
            assert g == pytest.approx(e)


def test_score_documents_skips_unknown_ids_and_preserves_corpus_stats(engine: SearchEngine):
    # Direct BM25 helper used by SearchEngine: deleted/unknown IDs ignored.
    ranker = BM25(k1=1.5, b=0.75, epsilon=0.25).fit(HAND_DOCS)
    ranked = ranker.score_documents("cat", ["d1", "missing", "d4", "d1"])
    assert [doc_id for doc_id, _ in ranked] == ["d1", "d4"]
    assert ranker.average_document_length == pytest.approx(2.8)
