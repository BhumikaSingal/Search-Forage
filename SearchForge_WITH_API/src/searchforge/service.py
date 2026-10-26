"""Persistent document operations coordinated with the in-memory search engine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import RLock

from searchforge.search import SearchEngine
from searchforge.storage import Document, DocumentSnapshot, SQLiteDocumentStore


class DuplicateDocumentError(Exception):
    """Raised when creating a document whose ID already exists."""


class DocumentNotFoundError(Exception):
    """Raised when updating or deleting a missing document."""


class SearchService:
    """Coordinate durable document mutations with a rebuilt SearchEngine."""

    def __init__(
        self,
        database_path: str | Path,
        initial_documents: Mapping[str, str] | None = None,
    ) -> None:
        self._store = SQLiteDocumentStore(database_path)
        self._lock = RLock()

        if initial_documents and not self._store.list_documents():
            seed_documents = candidate_documents(initial_documents)
            self._build_engine(seed_documents)
            with self._store.transaction() as transaction:
                for document in seed_documents:
                    transaction.create_document(document.id, document.text)

        self._index_data_version = self._store.data_version
        self._engine = self._build_engine(self._store.list_documents())
        self._index_needs_refresh = False

    @property
    def search_engine(self) -> SearchEngine:
        with self._lock:
            self._refresh_engine_if_stale()
            return self._engine

    @property
    def document_count(self) -> int:
        with self._lock:
            self._refresh_engine_if_stale()
            return self._engine.document_count

    @property
    def vocabulary_size(self) -> int:
        with self._lock:
            self._refresh_engine_if_stale()
            return self._engine.vocabulary_size

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            return self._store.get_document(doc_id)

    def list_documents(self) -> list[Document]:
        with self._lock:
            return self._store.list_documents()

    def list_documents_page(
        self, limit: int, offset: int
    ) -> tuple[int, list[Document]]:
        with self._lock:
            return self._store.list_documents_page(limit, offset)

    def create_document(self, doc_id: str, text: str) -> Document:
        self._validate_document(doc_id, text)
        with self._lock:
            commit_version = self._store.commit_version
            self._index_needs_refresh = True
            try:
                with self._store.transaction() as transaction:
                    if transaction.get_document(doc_id) is not None:
                        raise DuplicateDocumentError(doc_id)
                    transaction.create_document(doc_id, text)
                    candidate = self._build_engine(transaction.list_documents())
            except BaseException:
                if self._store.commit_version == commit_version:
                    self._index_needs_refresh = False
                raise
            self._engine = candidate
            self._index_needs_refresh = False
        return Document(id=doc_id, text=text)

    def update_document(self, doc_id: str, text: str) -> Document:
        self._validate_document(doc_id, text)
        with self._lock:
            commit_version = self._store.commit_version
            self._index_needs_refresh = True
            try:
                with self._store.transaction() as transaction:
                    if transaction.get_document(doc_id) is None:
                        raise DocumentNotFoundError(doc_id)
                    transaction.update_document(doc_id, text)
                    candidate = self._build_engine(transaction.list_documents())
            except BaseException:
                if self._store.commit_version == commit_version:
                    self._index_needs_refresh = False
                raise
            self._engine = candidate
            self._index_needs_refresh = False
        return Document(id=doc_id, text=text)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            commit_version = self._store.commit_version
            self._index_needs_refresh = True
            try:
                with self._store.transaction() as transaction:
                    if transaction.get_document(doc_id) is None:
                        raise DocumentNotFoundError(doc_id)
                    transaction.delete_document(doc_id)
                    candidate = self._build_engine(transaction.list_documents())
            except BaseException:
                if self._store.commit_version == commit_version:
                    self._index_needs_refresh = False
                raise
            self._engine = candidate
            self._index_needs_refresh = False

    def search(self, query: str) -> list[tuple[str, float]]:
        with self._lock:
            self._refresh_engine_if_stale()
            return self._engine.search(query)

    def search_documents(self, query: str) -> list[tuple[Document, float]]:
        with self._lock:
            with self._store.read_snapshot() as snapshot:
                data_version = self._store.data_version
                snapshot.establish()
                snapshot_data_version = self._store.data_version
                self._refresh_engine_if_stale(
                    snapshot, data_version, snapshot_data_version
                )
                ranked = self._engine.search(query)
                return [
                    (document, score)
                    for doc_id, score in ranked
                    if (document := snapshot.get_document(doc_id)) is not None
                ]

    def close(self) -> None:
        self._store.close()

    @staticmethod
    def _validate_document(doc_id: str, text: str) -> None:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document ID must not be empty")
        if not isinstance(text, str):
            raise TypeError("Document text must be a string")

    @staticmethod
    def _build_engine(documents: list[Document]) -> SearchEngine:
        engine = SearchEngine()
        engine.add_documents({document.id: document.text for document in documents})
        return engine

    def _refresh_engine_if_stale(
        self,
        snapshot: DocumentSnapshot | None = None,
        data_version: int | None = None,
        snapshot_data_version: int | None = None,
    ) -> None:
        if snapshot is None:
            with self._store.read_snapshot() as current_snapshot:
                current_data_version = self._store.data_version
                current_snapshot.establish()
                snapshot_data_version = self._store.data_version
                self._refresh_engine_if_stale(
                    current_snapshot,
                    current_data_version,
                    snapshot_data_version,
                )
            return

        if data_version is None:
            data_version = self._store.data_version
        if snapshot_data_version is None:
            snapshot_data_version = data_version
        if (
            not self._index_needs_refresh
            and data_version == self._index_data_version
            and snapshot_data_version == data_version
        ):
            return

        documents = snapshot.list_documents() if snapshot else self._store.list_documents()
        candidate = self._build_engine(documents)
        self._engine = candidate
        self._index_data_version = data_version
        self._index_needs_refresh = False


def candidate_documents(documents: Mapping[str, str]) -> list[Document]:
    return [
        Document(id=doc_id, text=text)
        for doc_id, text in sorted(documents.items())
    ]