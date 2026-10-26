"""Okapi BM25 ranking over an in-memory document collection.

Uses the ATIRE/Okapi IDF formulation with an epsilon floor for negative
IDF values, matching the widely used ``rank_bm25.BM25Okapi`` reference.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from searchforge.tokenizer import tokenize


class BM25:
    """Rank documents with Okapi BM25.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter (typical range 1.2–2.0).
    b:
        Length-normalization strength in ``[0, 1]`` (0 disables length norm).
    epsilon:
        Multiplier of average IDF used as a floor when raw IDF is negative
        (terms that appear in more than half the collection).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not math.isfinite(k1):
            raise ValueError("k1 must be finite")
        if not math.isfinite(b):
            raise ValueError("b must be finite")
        if not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite")
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1 inclusive")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self._doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._average_idf: float = 0.0
        self._doc_index: dict[str, int] = {}

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    @property
    def average_document_length(self) -> float:
        """Mean token count across indexed documents (0 if empty)."""
        return self._avgdl

    @property
    def document_ids(self) -> frozenset[str]:
        return frozenset(self._doc_ids)

    def fit(self, documents: Mapping[str, str]) -> BM25:
        """Index ``documents`` mapping doc_id → raw text. Replaces any prior corpus."""
        if not isinstance(documents, Mapping):
            raise TypeError("documents must be a mapping of doc_id to text")

        candidate = BM25(self.k1, self.b, self.epsilon)
        candidate._fit_documents(documents)
        self._doc_ids = candidate._doc_ids
        self._doc_len = candidate._doc_len
        self._tf = candidate._tf
        self._df = candidate._df
        self._idf = candidate._idf
        self._avgdl = candidate._avgdl
        self._average_idf = candidate._average_idf
        self._doc_index = candidate._doc_index
        return self

    def _fit_documents(self, documents: Mapping[str, str]) -> None:
        self._doc_ids = []
        self._doc_len = []
        self._tf = []
        self._df = {}
        self._idf = {}
        self._doc_index = {}
        self._avgdl = 0.0
        self._average_idf = 0.0

        total_tokens = 0
        for doc_id, text in documents.items():
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Document ID must not be empty")
            if not isinstance(text, str):
                raise TypeError("Document text must be a string")
            if doc_id in self._doc_index:
                raise ValueError(f"Duplicate document ID: {doc_id!r}")

            tokens = tokenize(text)
            counts = dict(Counter(tokens))
            length = len(tokens)

            self._doc_index[doc_id] = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            self._doc_len.append(length)
            self._tf.append(counts)
            total_tokens += length

            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1

        n_docs = len(self._doc_ids)
        self._avgdl = (total_tokens / n_docs) if n_docs else 0.0
        self._calc_idf()

    def document_frequency(self, term: str) -> int:
        """Return DF for a single normalized term; unknown / multi-token → 0."""
        tokens = tokenize(term)
        if len(tokens) != 1:
            return 0
        return self._df.get(tokens[0], 0)

    def term_frequency(self, doc_id: str, term: str) -> int:
        """Return TF of ``term`` in ``doc_id``; missing doc/term → 0."""
        idx = self._doc_index.get(doc_id)
        if idx is None:
            return 0
        tokens = tokenize(term)
        if len(tokens) != 1:
            return 0
        return self._tf[idx].get(tokens[0], 0)

    def document_length(self, doc_id: str) -> int | None:
        """Return token length of a document, or None if unknown."""
        idx = self._doc_index.get(doc_id)
        if idx is None:
            return None
        return self._doc_len[idx]

    def idf(self, term: str) -> float:
        """Return IDF for a single normalized term; unknown / multi-token → 0."""
        tokens = tokenize(term)
        if len(tokens) != 1:
            return 0.0
        return self._idf.get(tokens[0], 0.0)

    def score(self, query: str) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` pairs sorted by descending score, then doc_id.

        Empty or punctuation-only queries yield an empty list. Terms absent
        from the collection contribute nothing (IDF treated as 0).
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        terms = tokenize(query)
        if not terms or not self._doc_ids:
            return []

        scores = [0.0] * len(self._doc_ids)
        avgdl = self._avgdl

        for term in terms:
            term_idf = self._idf.get(term)
            if term_idf is None:
                continue

            for i, tf_map in enumerate(self._tf):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                scores[i] += term_idf * self._tf_norm(tf, self._doc_len[i], avgdl)

        ranked = sorted(
            zip(self._doc_ids, scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked

    def score_document(self, query: str, doc_id: str) -> float:
        """BM25 score of a single document for ``query`` (0 if missing/empty)."""
        idx = self._doc_index.get(doc_id)
        if idx is None:
            return 0.0

        terms = tokenize(query)
        if not terms:
            return 0.0

        return self._score_terms_at(terms, idx)

    def score_documents(
        self,
        query: str,
        doc_ids: Iterable[str],
    ) -> list[tuple[str, float]]:
        """Score a candidate set with corpus-wide IDF and average length.

        Unknown or deleted ``doc_ids`` are skipped. Empty queries yield ``[]``.
        Results are sorted by descending score, then ascending ``doc_id``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        terms = tokenize(query)
        if not terms or not self._doc_ids:
            return []

        seen: set[str] = set()
        scored: list[tuple[str, float]] = []
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            idx = self._doc_index.get(doc_id)
            if idx is None:
                continue
            seen.add(doc_id)
            scored.append((doc_id, self._score_terms_at(terms, idx)))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def _score_terms_at(self, terms: list[str], idx: int) -> float:
        """BM25 score for pre-tokenized ``terms`` at a corpus document index."""
        total = 0.0
        avgdl = self._avgdl
        tf_map = self._tf[idx]
        doc_len = self._doc_len[idx]

        for term in terms:
            term_idf = self._idf.get(term)
            if term_idf is None:
                continue
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            total += term_idf * self._tf_norm(tf, doc_len, avgdl)

        return total

    def _calc_idf(self) -> None:
        """ATIRE Okapi IDF with epsilon floor for negative weights."""
        n_docs = self.document_count
        if not self._df or n_docs == 0:
            self._idf = {}
            self._average_idf = 0.0
            return

        idf: dict[str, float] = {}
        negative_terms: list[str] = []
        idf_sum = 0.0

        for term, df in self._df.items():
            # log((N - df + 0.5) / (df + 0.5)); negative when df > N/2
            value = math.log(n_docs - df + 0.5) - math.log(df + 0.5)
            idf[term] = value
            idf_sum += value
            if value < 0:
                negative_terms.append(term)

        self._average_idf = idf_sum / len(idf)
        floor = self.epsilon * self._average_idf
        for term in negative_terms:
            idf[term] = floor

        self._idf = idf

    def _tf_norm(self, tf: int, doc_len: int, avgdl: float) -> float:
        """Saturated TF component with length normalization."""
        if avgdl == 0.0:
            # All documents empty: TF is always 0 in practice; avoid /0.
            length_norm = 1.0
        else:
            length_norm = 1.0 - self.b + self.b * (doc_len / avgdl)
        return (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
