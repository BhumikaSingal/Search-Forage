import os
import sqlite3
from contextlib import contextmanager
from multiprocessing import get_context

import pytest

from searchforge.service import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    SearchService,
)


def _exit_during_rebuild(database_path, operation):
    service = SearchService(database_path)
    service._build_engine = lambda _documents: os._exit(73)
    if operation == "create":
        service.create_document("d2", "interrupted replacement")
    elif operation == "update":
        service.update_document("d1", "interrupted replacement")
    else:
        service.delete_document("d1")


def _exit_after_commit_before_publish(database_path, operation):
    service = SearchService(database_path)
    original_transaction = service._store.transaction

    @contextmanager
    def commit_then_exit():
        with original_transaction() as transaction:
            yield transaction
        os._exit(74)

    service._store.transaction = commit_then_exit
    if operation == "create":
        service.create_document("d2", "committed replacement")
    elif operation == "update":
        service.update_document("d1", "committed replacement")
    else:
        service.delete_document("d1")


def _run_concurrent_mutations(database_path, worker_id, ready, start, results):
    service = SearchService(database_path)
    ready.put(worker_id)
    if not start.wait(timeout=15):
        results.put({"worker": worker_id, "error": "start signal timed out"})
        service.close()
        return

    try:
        service.create_document(
            f"created-{worker_id}", f"created worker-{worker_id} marker"
        )
        service.update_document(
            "shared-update", f"updated by worker-{worker_id}"
        )
        service.delete_document(f"delete-{worker_id}")
        try:
            service.create_document("race", f"race winner worker-{worker_id}")
            race_result = "created"
        except DuplicateDocumentError:
            race_result = "duplicate"
        results.put({"worker": worker_id, "race": race_result})
    except BaseException as error:
        results.put({"worker": worker_id, "error": repr(error)})
    finally:
        service.close()


def test_service_recovers_documents_and_ranker_after_restart(tmp_path):
    database_path = tmp_path / "searchforge.sqlite3"
    first = SearchService(database_path)
    first.create_document("d1", "persistent search term")
    first.close()

    recovered = SearchService(database_path)
    assert recovered.get_document("d1").text == "persistent search term"
    assert recovered.search("persistent")[0][0] == "d1"
    recovered.close()


def test_service_refreshes_search_index_after_external_connection_changes(tmp_path):
    database_path = tmp_path / "shared.sqlite3"
    writer = SearchService(database_path)
    observer = SearchService(database_path)

    try:
        writer.create_document("d1", "shared original token")
        assert [doc_id for doc_id, _ in observer.search("original")] == ["d1"]

        writer.update_document("d1", "shared replacement token")
        assert observer.search("original") == []
        assert [doc_id for doc_id, _ in observer.search("replacement")] == ["d1"]

        writer.delete_document("d1")
        assert observer.search("replacement") == []
    finally:
        writer.close()
        observer.close()

    recovered = SearchService(database_path)
    assert recovered.list_documents() == []
    assert recovered.search("replacement") == []
    recovered.close()


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_search_uses_one_snapshot_during_external_mutation(
    tmp_path, monkeypatch, operation
):
    database_path = tmp_path / f"snapshot-{operation}.sqlite3"
    writer = SearchService(database_path)
    writer.create_document("d1", "oldterm original text")
    reader = SearchService(database_path)
    engine = reader.search_engine
    original_search = engine.search

    def mutate_after_ranking(query):
        ranked = original_search(query)
        if operation == "update":
            writer.update_document("d1", "newterm replacement text")
        else:
            writer.delete_document("d1")
        return ranked

    monkeypatch.setattr(engine, "search", mutate_after_ranking)
    try:
        first_response = reader.search_documents("oldterm")
        assert [(document.id, document.text) for document, _ in first_response] == [
            ("d1", "oldterm original text")
        ]

        assert reader.search_documents("oldterm") == []
        if operation == "update":
            updated = reader.search_documents("newterm")
            assert [(document.id, document.text) for document, _ in updated] == [
                ("d1", "newterm replacement text")
            ]
        else:
            assert reader.search_documents("newterm") == []
    finally:
        reader.close()
        writer.close()


def test_service_mutations_rebuild_postings_without_stale_terms(tmp_path):
    service = SearchService(tmp_path / "searchforge.sqlite3")
    service.create_document("d1", "old unique token")

    service.update_document("d1", "new unique token")
    assert service.search("old") == []
    assert service.search("new")[0][0] == "d1"

    service.delete_document("d1")
    assert service.search("new") == []
    service.close()


def test_service_rejects_duplicate_create_and_missing_mutations(tmp_path):
    service = SearchService(tmp_path / "searchforge.sqlite3")
    service.create_document("d1", "stored")

    with pytest.raises(DuplicateDocumentError):
        service.create_document("d1", "duplicate")

    mutations = (
        lambda: service.update_document("missing", "text"),
        lambda: service.delete_document("missing"),
    )
    for mutation in mutations:
        with pytest.raises(DocumentNotFoundError):
            mutation()

    assert service.get_document("d1").text == "stored"
    service.close()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_failed_rebuild_rolls_back_persistence_and_keeps_live_index(
    tmp_path, monkeypatch, operation
):
    database_path = tmp_path / "searchforge.sqlite3"
    service = SearchService(database_path)
    service.create_document("d1", "committed original")

    def fail_rebuild(_documents):
        raise RuntimeError("simulated rebuild failure")

    monkeypatch.setattr(service, "_build_engine", fail_rebuild)
    with pytest.raises(RuntimeError, match="simulated rebuild failure"):
        if operation == "create":
            service.create_document("d2", "uncommitted replacement")
        elif operation == "update":
            service.update_document("d1", "uncommitted replacement")
        else:
            service.delete_document("d1")

    assert [(doc.id, doc.text) for doc in service.list_documents()] == [
        ("d1", "committed original")
    ]
    assert [doc_id for doc_id, _ in service.search("committed")] == ["d1"]
    assert service.search("replacement") == []
    service.close()

    recovered = SearchService(database_path)
    assert [(doc.id, doc.text) for doc in recovered.list_documents()] == [
        ("d1", "committed original")
    ]
    assert [doc_id for doc_id, _ in recovered.search("committed")] == ["d1"]
    assert recovered.search("replacement") == []
    recovered.close()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_sql_write_failure_preserves_store_and_index_after_restart(tmp_path, operation):
    database_path = tmp_path / "write-failure.sqlite3"
    service = SearchService(database_path)
    service.create_document("d1", "committed original")

    if operation == "create":
        trigger = (
            "CREATE TRIGGER injected_write_failure BEFORE INSERT ON documents "
            "WHEN NEW.id = 'd2' BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"
        )
        mutation = lambda: service.create_document("d2", "replacement")
    elif operation == "update":
        trigger = (
            "CREATE TRIGGER injected_write_failure BEFORE UPDATE ON documents "
            "WHEN OLD.id = 'd1' BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"
        )
        mutation = lambda: service.update_document("d1", "replacement")
    else:
        trigger = (
            "CREATE TRIGGER injected_write_failure BEFORE DELETE ON documents "
            "WHEN OLD.id = 'd1' BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"
        )
        mutation = lambda: service.delete_document("d1")

    service._store._connection.execute(trigger)
    service._store._connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
        mutation()

    assert [(doc.id, doc.text) for doc in service.list_documents()] == [
        ("d1", "committed original")
    ]
    assert [doc_id for doc_id, _ in service.search("committed")] == ["d1"]
    assert service.search("replacement") == []
    service.close()

    recovered = SearchService(database_path)
    assert [(doc.id, doc.text) for doc in recovered.list_documents()] == [
        ("d1", "committed original")
    ]
    assert [doc_id for doc_id, _ in recovered.search("committed")] == ["d1"]
    assert recovered.search("replacement") == []
    recovered.close()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_post_commit_exception_refreshes_live_index_and_restart(
    tmp_path, monkeypatch, operation
):
    database_path = tmp_path / "post-commit.sqlite3"
    service = SearchService(database_path)
    service.create_document("d1", "baseline text")
    original_transaction = service._store.transaction

    @contextmanager
    def commit_then_fail():
        with original_transaction() as transaction:
            yield transaction
        raise RuntimeError("injected after commit")

    monkeypatch.setattr(service._store, "transaction", commit_then_fail)
    with pytest.raises(RuntimeError, match="injected after commit"):
        if operation == "create":
            service.create_document("d2", "committed change")
        elif operation == "update":
            service.update_document("d1", "committed change")
        else:
            service.delete_document("d1")

    expected_documents = {
        "create": {"d1": "baseline text", "d2": "committed change"},
        "update": {"d1": "committed change"},
        "delete": {},
    }[operation]
    expected_committed_ids = {
        "create": {"d2"},
        "update": {"d1"},
        "delete": set(),
    }[operation]
    expected_baseline_ids = {"d1"} if operation == "create" else set()
    assert {
        document.id: document.text for document in service.list_documents()
    } == expected_documents
    assert {doc_id for doc_id, _ in service.search("committed")} == (
        expected_committed_ids
    )
    assert {doc_id for doc_id, _ in service.search("baseline")} == (
        expected_baseline_ids
    )
    service.close()

    recovered = SearchService(database_path)
    assert {
        document.id: document.text for document in recovered.list_documents()
    } == expected_documents
    assert {doc_id for doc_id, _ in recovered.search("committed")} == (
        expected_committed_ids
    )
    assert {doc_id for doc_id, _ in recovered.search("baseline")} == (
        expected_baseline_ids
    )
    recovered.close()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_process_exit_after_write_before_commit_recovers_previous_state(
    tmp_path, operation
):
    database_path = tmp_path / "interrupted.sqlite3"
    service = SearchService(database_path)
    service.create_document("d1", "committed original")
    service.close()

    process = get_context("spawn").Process(
        target=_exit_during_rebuild, args=(str(database_path), operation)
    )
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("fault-injection worker did not exit")
    assert process.exitcode == 73

    recovered = SearchService(database_path)
    assert [(doc.id, doc.text) for doc in recovered.list_documents()] == [
        ("d1", "committed original")
    ]
    assert [doc_id for doc_id, _ in recovered.search("committed")] == ["d1"]
    assert recovered.search("replacement") == []
    recovered.close()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_process_exit_after_commit_recovers_committed_state(tmp_path, operation):
    database_path = tmp_path / "committed-interruption.sqlite3"
    service = SearchService(database_path)
    service.create_document("d1", "original baseline")
    service.close()

    process = get_context("spawn").Process(
        target=_exit_after_commit_before_publish,
        args=(str(database_path), operation),
    )
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("fault-injection worker did not exit")
    assert process.exitcode == 74

    recovered = SearchService(database_path)
    expected_documents = {
        "create": {"d1": "original baseline", "d2": "committed replacement"},
        "update": {"d1": "committed replacement"},
        "delete": {},
    }[operation]
    assert {
        document.id: document.text for document in recovered.list_documents()
    } == expected_documents
    assert {doc_id for doc_id, _ in recovered.search("replacement")} == (
        {"d2"} if operation == "create" else {"d1"} if operation == "update" else set()
    )
    assert {doc_id for doc_id, _ in recovered.search("baseline")} == (
        {"d1"} if operation == "create" else set()
    )
    recovered.close()


def test_separate_processes_serialize_crud_and_recover_shared_state(tmp_path):
    database_path = tmp_path / "multiprocess.sqlite3"
    initializer = SearchService(database_path)
    initializer.create_document("shared-update", "original shared text")
    initializer.create_document("delete-0", "toberemoved worker zero")
    initializer.create_document("delete-1", "toberemoved worker one")
    initializer.close()

    observer = SearchService(database_path)
    context = get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_run_concurrent_mutations,
            args=(str(database_path), worker_id, ready, start, results),
        )
        for worker_id in range(2)
    ]

    try:
        for process in processes:
            process.start()
        assert {ready.get(timeout=20) for _ in processes} == {0, 1}
        start.set()

        worker_results = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert not process.is_alive(), "mutation worker did not exit"
            assert process.exitcode == 0

        assert all("error" not in result for result in worker_results)
        assert sorted(result["race"] for result in worker_results) == [
            "created",
            "duplicate",
        ]
        assert {doc_id for doc_id, _ in observer.search("marker")} == {
            "created-0",
            "created-1",
        }
        assert {doc_id for doc_id, _ in observer.search("updated")} == {
            "shared-update"
        }
        assert observer.search("toberemoved") == []
        assert observer.document_count == 4
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        observer.close()
        ready.close()
        results.close()

    recovered = SearchService(database_path)
    documents = {document.id: document.text for document in recovered.list_documents()}
    assert set(documents) == {"created-0", "created-1", "race", "shared-update"}
    assert documents["shared-update"] in {
        "updated by worker-0",
        "updated by worker-1",
    }
    assert {doc_id for doc_id, _ in recovered.search("marker")} == {
        "created-0",
        "created-1",
    }
    assert recovered.search("original") == []
    assert recovered.search("toberemoved") == []
    assert [doc_id for doc_id, _ in recovered.search("winner")] == ["race"]
    recovered.close()