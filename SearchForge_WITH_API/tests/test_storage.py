import os
import shutil
import sqlite3
import stat
from pathlib import Path

import pytest

from searchforge.storage import SQLiteDocumentStore


def test_file_store_uses_wal_and_full_synchronous_mode(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "settings.sqlite3")
    try:
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store._connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        store.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not available")
def test_new_database_and_parent_directory_are_owner_only(tmp_path):
    database_path = tmp_path / "private" / "nested" / "documents.sqlite3"
    store = SQLiteDocumentStore(database_path)
    try:
        with store.transaction() as transaction:
            transaction.create_document("d1", "private data")
        assert stat.S_IMODE(database_path.parent.stat().st_mode) & 0o077 == 0
        assert stat.S_IMODE(database_path.parent.parent.stat().st_mode) & 0o077 == 0
        assert stat.S_IMODE(database_path.stat().st_mode) & 0o077 == 0

        for sidecar in (Path(f"{database_path}-wal"), Path(f"{database_path}-shm")):
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) & 0o077 == 0
    finally:
        store.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not available")
def test_existing_database_permissions_are_not_changed(tmp_path):
    database_path = tmp_path / "existing.sqlite3"
    database_path.touch(mode=0o640)
    database_path.chmod(0o640)

    store = SQLiteDocumentStore(database_path)
    try:
        assert stat.S_IMODE(database_path.stat().st_mode) == 0o640
    finally:
        store.close()


def _make_wal_copy(tmp_path: Path, corruption: str) -> Path:
    source = tmp_path / "source.sqlite3"
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute(
        "CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
    )
    connection.commit()
    connection.execute(
        "INSERT INTO documents (id, text) VALUES (?, ?)",
        ("checkpointed", "retained base data"),
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute(
        "INSERT INTO documents (id, text) VALUES (?, ?)",
        ("wal-only", "committed in WAL"),
    )
    connection.commit()

    source_wal = Path(f"{source}-wal")
    wal_bytes = source_wal.read_bytes()
    clone = tmp_path / f"{corruption}.sqlite3"
    shutil.copyfile(source, clone)

    if corruption == "truncated":
        damaged_wal = wal_bytes[:-20]
    elif corruption == "checksum":
        damaged_wal = wal_bytes[:-1] + bytes([wal_bytes[-1] ^ 0xFF])
    else:
        damaged_wal = bytes([wal_bytes[0] ^ 0xFF]) + wal_bytes[1:]

    Path(f"{clone}-wal").write_bytes(damaged_wal)
    connection.close()
    return clone


@pytest.mark.parametrize("corruption", ["truncated", "checksum", "header"])
def test_sqlite_recovery_retains_checkpointed_rows_for_damaged_wal_tail(
    tmp_path, corruption
):
    database_path = _make_wal_copy(tmp_path, corruption)
    store = SQLiteDocumentStore(database_path)
    try:
        rows = [(document.id, document.text) for document in store.list_documents()]
        integrity = store._connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        store.close()

    assert rows == [("checkpointed", "retained base data")]
    assert integrity == "ok"