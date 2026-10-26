"""SQLite storage for SearchForge documents."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterator


@dataclass(frozen=True)
class Document:
    id: str
    text: str


class DocumentTransaction:
    """Document operations scoped to one SQLite transaction."""

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    def get_document(self, doc_id: str) -> Document | None:
        row = self._cursor.execute(
            "SELECT id, text FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(id=row["id"], text=row["text"])

    def list_documents(self) -> list[Document]:
        rows = self._cursor.execute(
            "SELECT id, text FROM documents ORDER BY id"
        ).fetchall()
        return [Document(id=row["id"], text=row["text"]) for row in rows]

    def create_document(self, doc_id: str, text: str) -> None:
        self._cursor.execute(
            "INSERT INTO documents (id, text) VALUES (?, ?)", (doc_id, text)
        )

    def update_document(self, doc_id: str, text: str) -> None:
        self._cursor.execute(
            "UPDATE documents SET text = ? WHERE id = ?", (text, doc_id)
        )

    def delete_document(self, doc_id: str) -> None:
        self._cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


class DocumentSnapshot:
    """Read-only document queries pinned to one SQLite transaction snapshot."""

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    def establish(self) -> None:
        self._cursor.execute("SELECT COUNT(*) FROM documents").fetchone()

    def get_document(self, doc_id: str) -> Document | None:
        row = self._cursor.execute(
            "SELECT id, text FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(id=row["id"], text=row["text"])

    def list_documents(self) -> list[Document]:
        rows = self._cursor.execute(
            "SELECT id, text FROM documents ORDER BY id"
        ).fetchall()
        return [Document(id=row["id"], text=row["text"]) for row in rows]

    def list_documents_page(
        self, limit: int, offset: int
    ) -> tuple[int, list[Document]]:
        total = int(
            self._cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        rows = self._cursor.execute(
            "SELECT id, text FROM documents ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return total, [Document(id=row["id"], text=row["text"]) for row in rows]


class SQLiteDocumentStore:
    """Thread-safe SQLite connection with explicit write transactions."""

    def __init__(self, database_path: str | Path) -> None:
        path = str(database_path)
        if path != ":memory:":
            database_file = Path(path).expanduser()
            _create_private_directories(database_file.parent)
            try:
                file_descriptor = os.open(
                    database_file,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                    0o600,
                )
            except FileExistsError:
                pass
            else:
                os.close(file_descriptor)
            path = str(database_file)

        self._lock = RLock()
        self._commit_version = 0
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "id TEXT PRIMARY KEY NOT NULL, "
            "text TEXT NOT NULL)"
        )
        self._connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[DocumentTransaction]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield DocumentTransaction(self._connection.cursor())
                self._connection.commit()
                self._commit_version += 1
            except BaseException:
                self._connection.rollback()
                raise

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, text FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        return Document(id=row["id"], text=row["text"])

    def list_documents(self) -> list[Document]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, text FROM documents ORDER BY id"
            ).fetchall()
        return [Document(id=row["id"], text=row["text"]) for row in rows]

    @contextmanager
    def read_snapshot(self) -> Iterator[DocumentSnapshot]:
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                yield DocumentSnapshot(self._connection.cursor())
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def list_documents_page(
        self, limit: int, offset: int
    ) -> tuple[int, list[Document]]:
        with self.read_snapshot() as snapshot:
            return snapshot.list_documents_page(limit, offset)

    @property
    def data_version(self) -> int:
        """Return SQLite's connection-local signal for commits by other connections."""
        with self._lock:
            row = self._connection.execute("PRAGMA data_version").fetchone()
        return int(row[0])

    @property
    def commit_version(self) -> int:
        """Return the number of successful writes committed by this connection."""
        with self._lock:
            return self._commit_version

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _create_private_directories(directory: Path) -> None:
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent

    for path in reversed(missing):
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            if not path.is_dir():
                raise