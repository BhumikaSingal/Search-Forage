"""Tests for the inverted index."""

import pytest

from searchforge.index import InvertedIndex


@pytest.fixture
def index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.add_document("doc1", "The cat sat on the mat.")
    idx.add_document("doc2", "Dogs and cats are friends.")
    idx.add_document("doc3", "The dog chased the cat.")
    return idx


# --- Adding documents ---


def test_empty_index_has_no_documents():
    idx = InvertedIndex()
    assert idx.document_count == 0
    assert idx.vocabulary_size == 0
    assert idx.lookup("cat") == frozenset()


def test_add_document_increases_counts(index: InvertedIndex):
    assert index.document_count == 3
    assert index.vocabulary_size > 0
    assert index.document_ids == frozenset({"doc1", "doc2", "doc3"})


def test_add_document_rejects_empty_doc_id():
    idx = InvertedIndex()
    with pytest.raises(ValueError):
        idx.add_document("", "hello")


def test_empty_document_is_stored_but_adds_no_terms():
    idx = InvertedIndex()
    idx.add_document("empty", "")
    assert idx.document_count == 1
    assert idx.vocabulary_size == 0
    assert idx.get_document("empty") == ""


def test_readding_document_replaces_old_content():
    idx = InvertedIndex()
    idx.add_document("doc1", "alpha beta")
    idx.add_document("doc1", "beta gamma")

    assert idx.lookup("alpha") == frozenset()
    assert idx.lookup("beta") == frozenset({"doc1"})
    assert idx.lookup("gamma") == frozenset({"doc1"})
    assert idx.get_document("doc1") == "beta gamma"


# --- Lookup (single term) ---


def test_lookup_finds_documents_containing_term(index: InvertedIndex):
    # No stemming yet: "cat" and "cats" are different terms
    assert index.lookup("cat") == frozenset({"doc1", "doc3"})
    assert index.lookup("cats") == frozenset({"doc2"})
    assert index.lookup("dog") == frozenset({"doc3"})
    assert index.lookup("dogs") == frozenset({"doc2"})
    assert index.lookup("mat") == frozenset({"doc1"})


def test_lookup_unknown_term_returns_empty(index: InvertedIndex):
    assert index.lookup("zebra") == frozenset()


def test_lookup_is_case_and_punctuation_insensitive(index: InvertedIndex):
    assert index.lookup("CAT") == index.lookup("cat")
    assert index.lookup("cat!") == index.lookup("cat")


def test_lookup_rejects_multi_word_and_empty_input(index: InvertedIndex):
    assert index.lookup("cat dog") == frozenset()
    assert index.lookup("") == frozenset()
    assert index.lookup("...") == frozenset()


# --- Search (AND of query terms) ---


def test_search_single_term_matches_lookup(index: InvertedIndex):
    assert index.search("cat") == index.lookup("cat")


def test_search_and_intersection(index: InvertedIndex):
    # Only doc3 has both "dog" and "cat"
    assert index.search("dog cat") == frozenset({"doc3"})


def test_search_no_overlap_returns_empty(index: InvertedIndex):
    assert index.search("mat dogs") == frozenset()


def test_search_empty_query_returns_empty(index: InvertedIndex):
    assert index.search("") == frozenset()
    assert index.search("   ") == frozenset()


def test_search_duplicate_query_terms(index: InvertedIndex):
    assert index.search("cat cat") == index.lookup("cat")


# --- Unicode ---


def test_index_and_lookup_unicode():
    idx = InvertedIndex()
    idx.add_document("fr", "Café résumé")
    assert idx.lookup("café") == frozenset({"fr"})
    assert idx.lookup("CAFÉ") == frozenset({"fr"})
    assert idx.search("café résumé") == frozenset({"fr"})


# --- Document retrieval ---


def test_get_document_returns_original_text(index: InvertedIndex):
    assert index.get_document("doc1") == "The cat sat on the mat."
    assert index.get_document("missing") is None


# --- Delete / candidates (OR retrieval) ---


def test_delete_document_removes_postings_without_stale_entries():
    idx = InvertedIndex()
    idx.add_document("a", "cat dog")
    idx.add_document("b", "cat bird")
    assert idx.delete_document("a") is True
    assert idx.delete_document("a") is False
    assert idx.lookup("dog") == frozenset()
    assert idx.lookup("cat") == frozenset({"b"})
    assert idx.document_count == 1
    assert idx.vocabulary_size == 2  # cat, bird — dog posting removed


def test_candidates_are_union_of_query_term_postings(index: InvertedIndex):
    # OR of terms: mat ∪ dogs
    assert index.candidates("mat dogs") == frozenset({"doc1", "doc2"})
    assert index.candidates("") == frozenset()
    assert index.candidates("zebra") == frozenset()
    # Token-less document is stored but never a candidate
    index.add_document("empty", "")
    assert index.candidates("cat") == frozenset({"doc1", "doc3"})
    assert "empty" not in index.candidates("cat")


def test_documents_view_is_read_only(index: InvertedIndex):
    view = index.documents
    assert view["doc1"] == "The cat sat on the mat."
    with pytest.raises(TypeError):
        view["doc1"] = "mutated"  # type: ignore[index]
