"""In-memory inverted index for document retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from types import MappingProxyType

from searchforge.tokenizer import tokenize


class InvertedIndex:
    """Map normalized terms to the document IDs that contain them."""

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    @property
    def document_ids(self) -> frozenset[str]:
        return frozenset(self._documents)

    @property
    def documents(self) -> Mapping[str, str]:
        """Read-only view of doc_id → original text."""
        return MappingProxyType(self._documents)

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document, replacing any existing document with the same ID."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document ID must not be empty")

        if not isinstance(text, str):
            raise TypeError("Document text must be a string")

        # Remove old postings before replacing existing document content.
        if doc_id in self._documents:
            self._remove_postings(doc_id, self._documents[doc_id])

        self._documents[doc_id] = text

        for term in set(tokenize(text)):
            self._postings[term].add(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and its postings. Return True if it existed."""
        text = self._documents.pop(doc_id, None)
        if text is None:
            return False
        self._remove_postings(doc_id, text)
        return True

    def lookup(self, term: str) -> frozenset[str]:
        """Return document IDs containing exactly one normalized term."""
        tokens = tokenize(term)
        if len(tokens) != 1:
            return frozenset()

        return frozenset(self._postings.get(tokens[0], set()))

    def search(self, query: str) -> frozenset[str]:
        """Return documents containing every distinct term in the query."""
        terms = tokenize(query)
        if not terms:
            return frozenset()

        result = self.lookup(terms[0])
        for term in terms[1:]:
            result = result.intersection(self.lookup(term))
            if not result:
                break

        return frozenset(result)

    def candidates(self, query: str) -> frozenset[str]:
        """Return documents containing at least one known query term (OR).

        Empty / punctuation-only queries and queries whose terms are all
        absent from the vocabulary yield an empty set. Token-less documents
        never appear because they have no postings.
        """
        terms = tokenize(query)
        if not terms:
            return frozenset()

        found: set[str] = set()
        for term in terms:
            postings = self._postings.get(term)
            if postings:
                found.update(postings)
        return frozenset(found)

    def get_document(self, doc_id: str) -> str | None:
        """Return the original document text, or None if it does not exist."""
        return self._documents.get(doc_id)

    def _remove_postings(self, doc_id: str, text: str) -> None:
        for term in set(tokenize(text)):
            bucket = self._postings.get(term)
            if bucket is None:
                continue
            bucket.discard(doc_id)
            if not bucket:
                del self._postings[term]
