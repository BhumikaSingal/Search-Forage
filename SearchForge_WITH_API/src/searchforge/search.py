"""End-to-end search: tokenize → inverted-index candidates → BM25 ranking."""

from __future__ import annotations

from collections.abc import Mapping

from searchforge.bm25 import BM25
from searchforge.index import InvertedIndex
from searchforge.tokenizer import tokenize


class SearchEngine:
    """Incremental document store with inverted-index retrieval and BM25 ranking.

    Corpus statistics for BM25 (document frequency, average length, ``N``) are
    always computed over the full indexed collection, including documents that
    have no searchable tokens. Candidate generation uses the inverted index
    (OR of query-term postings) so ranking does not scan the entire corpus.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self._index = InvertedIndex()
        self._ranker = BM25(k1=k1, b=b, epsilon=epsilon)

    @property
    def document_count(self) -> int:
        return self._index.document_count

    @property
    def vocabulary_size(self) -> int:
        return self._index.vocabulary_size

    @property
    def average_document_length(self) -> float:
        return self._ranker.average_document_length

    @property
    def document_ids(self) -> frozenset[str]:
        return self._index.document_ids

    def add_document(self, doc_id: str, text: str) -> None:
        """Insert or replace a document and refresh corpus-wide BM25 statistics."""
        self._index.add_document(doc_id, text)
        self._rebuild_ranker()

    def add_documents(self, documents: Mapping[str, str]) -> None:
        """Insert or replace documents, refreshing corpus statistics once."""
        if not documents:
            return
        for doc_id, text in documents.items():
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Document ID must not be empty")
            if not isinstance(text, str):
                raise TypeError("Document text must be a string")

        for doc_id, text in documents.items():
            self._index.add_document(doc_id, text)
        self._rebuild_ranker()

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and refresh statistics. Return True if it existed."""
        deleted = self._index.delete_document(doc_id)
        if deleted:
            self._rebuild_ranker()
        return deleted

    def get_document(self, doc_id: str) -> str | None:
        return self._index.get_document(doc_id)

    def document_frequency(self, term: str) -> int:
        return self._ranker.document_frequency(term)

    def search(self, query: str, *, limit: int | None = None) -> list[tuple[str, float]]:
        """Rank documents for ``query`` as ``(doc_id, score)`` pairs.

        Behavior
        --------
        - Empty / punctuation-only queries → ``[]``.
        - Unknown terms are ignored; if every term is unknown → ``[]``.
        - Candidates are the union of inverted-index postings for known terms.
        - Scores use corpus-wide DF / avgdl (not candidate-local statistics).
        - Deleted documents never appear.
        - Documents with no searchable tokens are kept in corpus stats but are
          not candidates (they have no postings).
        - Order is deterministic: descending score, then ascending ``doc_id``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        terms = tokenize(query)
        if not terms or self._index.document_count == 0:
            return []

        candidates = self._index.candidates(query)
        if not candidates:
            return []

        ranked = self._ranker.score_documents(query, candidates)
        if limit is None:
            return ranked
        return ranked[:limit]

    def _rebuild_ranker(self) -> None:
        """Refit BM25 on the live document map so DF/avgdl stay consistent."""
        self._ranker.fit(dict(self._index.documents))
