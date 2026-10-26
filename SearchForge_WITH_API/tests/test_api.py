"""HTTP contract tests for the FastAPI layer."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from searchforge.api import create_app
from searchforge.search import SearchEngine


def make_client() -> TestClient:
    engine = SearchEngine()
    engine.add_document("d1", "cat sat on mat")
    engine.add_document("d2", "cat chased dog")
    engine.add_document("d3", "birds fly")
    return TestClient(create_app(engine))


def test_health_reports_live_index_stats():
    client = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["document_count"] == 3
    assert body["vocabulary_size"] > 0


def test_create_document_and_duplicate_conflict():
    client = make_client()
    created = client.post("/documents", json={"id": "d4", "text": "search engine"})
    assert created.status_code == 201
    assert created.json() == {"id": "d4", "text": "search engine"}

    duplicate = client.post("/documents", json={"id": "d4", "text": "replacement"})
    assert duplicate.status_code == 409


def test_document_list_and_get():
    client = make_client()
    response = client.get("/documents?limit=2&offset=1")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert [item["id"] for item in body["documents"]] == ["d2", "d3"]

    single = client.get("/documents/d1")
    assert single.status_code == 200
    assert single.json() == {"id": "d1", "text": "cat sat on mat"}
    assert client.get("/documents/missing").status_code == 404


def test_document_listing_empty_collection_and_final_page(tmp_path, monkeypatch):
    app = create_app(database_path=tmp_path / "empty-list.sqlite3")
    with TestClient(app) as client:
        empty = client.get("/documents")
        assert empty.status_code == 200
        assert empty.json() == {
            "total": 0,
            "limit": 20,
            "offset": 0,
            "documents": [],
        }

        for doc_id in ("d1", "d2", "d3"):
            assert client.post(
                "/documents", json={"id": doc_id, "text": doc_id}
            ).status_code == 201

        monkeypatch.setattr(
            app.state.search_service._store,
            "list_documents",
            lambda: pytest.fail("list endpoint must use SQL pagination"),
        )
        final_page = client.get("/documents", params={"limit": 2, "offset": 2})
        assert final_page.status_code == 200
        assert final_page.json()["total"] == 3
        assert [item["id"] for item in final_page.json()["documents"]] == ["d3"]

        beyond_end = client.get("/documents", params={"limit": 2, "offset": 10_000})
        assert beyond_end.status_code == 200
        assert beyond_end.json()["total"] == 3
        assert beyond_end.json()["documents"] == []


def test_put_updates_existing_document_and_rebuilds_ranking():
    client = make_client()
    missing = client.put("/documents/d4", json={"id": "d4", "text": "cat kitten"})
    assert missing.status_code == 404

    created = client.post("/documents", json={"id": "d4", "text": "cat kitten"})
    assert created.status_code == 201

    assert client.get("/search?q=kitten").json()["results"][0]["id"] == "d4"

    updated = client.put("/documents/d4", json={"id": "d4", "text": "unrelated words"})
    assert updated.status_code == 200
    assert updated.json() == {"id": "d4", "text": "unrelated words"}
    assert client.get("/search?q=kitten").json()["total"] == 0
    assert client.get("/search?q=unrelated").json()["results"][0]["id"] == "d4"
    assert client.get("/search?q=cat").json()["total"] == 2


def test_put_rejects_mismatched_path_and_body_ids():
    client = make_client()
    response = client.put("/documents/d1", json={"id": "d2", "text": "replacement"})
    assert response.status_code == 422


def test_delete_document_and_missing_delete():
    client = make_client()
    response = client.delete("/documents/d1")
    assert response.status_code == 204
    assert client.get("/documents/d1").status_code == 404
    assert client.get("/search?q=mat").json()["total"] == 0
    assert client.delete("/documents/d1").status_code == 404


def test_search_returns_ranked_results_and_paginates_after_ranking():
    client = make_client()
    first = client.get("/search", params={"q": "cat", "limit": 1})
    assert first.status_code == 200
    body = first.json()
    assert body["total"] == 2
    assert len(body["results"]) == 1
    assert body["results"][0]["id"] == "d2"

    second = client.get("/search", params={"q": "cat", "limit": 1, "offset": 1})
    assert second.status_code == 200
    assert second.json()["results"][0]["id"] == "d1"


def test_search_empty_and_unknown_query_return_empty_results():
    client = make_client()
    for query in ("", "   ", "unknownterm"):
        response = client.get("/search", params={"q": query})
        assert response.status_code == 200
        assert response.json()["total"] == 0
        assert response.json()["results"] == []
    assert client.get("/search").status_code == 422


def test_pagination_validation():
    client = make_client()
    assert client.get("/search?q=cat&limit=0").status_code == 422
    assert client.get("/search?q=cat&limit=101").status_code == 422
    assert client.get("/documents?limit=101").status_code == 422
    assert client.get("/documents?offset=-1").status_code == 422
    assert client.get("/documents?offset=10001").status_code == 422
    assert client.get("/search?q=cat&offset=10001").status_code == 422

    for response in (
        client.get("/documents?offset=10001"),
        client.get("/search?q=cat&offset=10001"),
    ):
        assert response.json()["error"]["code"] == "validation_error"


def test_search_requires_query_and_enforces_maximum_length():
    client = make_client()
    assert client.get("/search").status_code == 422

    response = client.get("/search", params={"q": "x" * 2_001})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_document_payload_validation():
    client = make_client()
    blank_id = client.post("/documents", json={"id": "  ", "text": "x"})
    invalid_text = client.post("/documents", json={"id": "new", "text": 123})
    assert blank_id.status_code == 422
    assert invalid_text.status_code == 422
    assert blank_id.json()["error"]["code"] == "validation_error"


def test_restart_recovers_persisted_documents_and_search_index(tmp_path):
    database_path = tmp_path / "searchforge.sqlite3"
    with TestClient(create_app(database_path=database_path)) as client:
        assert client.post(
            "/documents", json={"id": "persisted", "text": "durable search"}
        ).status_code == 201

    with TestClient(create_app(database_path=database_path)) as restarted:
        document = restarted.get("/documents/persisted")
        assert document.status_code == 200
        assert document.json()["text"] == "durable search"
        assert restarted.get("/search?q=durable").json()["results"][0]["id"] == "persisted"


def test_missing_document_errors_have_consistent_shape():
    client = make_client()
    response = client.get("/documents/missing")
    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "Document not found"}
    }


def test_unexpected_error_response_does_not_expose_exception_details(monkeypatch):
    app = create_app(database_path=":memory:")

    def fail_search(_query):
        raise RuntimeError("private database details")

    monkeypatch.setattr(app.state.search_service, "search_documents", fail_search)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/search", params={"q": "secret"})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected error occurred",
        }
    }
    assert "private database details" not in response.text


def test_concurrent_search_and_mutation_requests_remain_consistent(tmp_path):
    database_path = tmp_path / "concurrent.sqlite3"
    app = create_app(database_path=database_path)

    with TestClient(app) as client:
        for index in range(4):
            response = client.post(
                "/documents",
                json={"id": f"seed-{index}", "text": "shared seed content"},
            )
            assert response.status_code == 201

        with ThreadPoolExecutor(max_workers=8) as executor:
            search_responses = list(
                executor.map(
                    lambda _: client.get("/search", params={"q": "shared"}),
                    range(24),
                )
            )
        assert all(response.status_code == 200 for response in search_responses)
        assert {response.json()["total"] for response in search_responses} == {4}

        def create(index: int):
            return client.post(
                "/documents",
                json={"id": f"concurrent-{index:02d}", "text": "new shared text"},
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            create_responses = list(executor.map(create, range(16)))
        assert all(response.status_code == 201 for response in create_responses)

        def update(index: int):
            doc_id = f"concurrent-{index:02d}"
            return client.put(
                f"/documents/{doc_id}",
                json={"id": doc_id, "text": "updated shared text"},
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            update_responses = list(executor.map(update, range(16)))
        assert all(response.status_code == 200 for response in update_responses)

        def delete(index: int):
            return client.delete(f"/documents/concurrent-{index:02d}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            delete_responses = list(executor.map(delete, range(0, 16, 2)))
        assert all(response.status_code == 204 for response in delete_responses)

        health = client.get("/health")
        assert health.json()["document_count"] == 12
        assert client.get("/search?q=updated").json()["total"] == 8

    with TestClient(create_app(database_path=database_path)) as restarted:
        assert restarted.get("/health").json()["document_count"] == 12
        assert restarted.get("/search?q=updated").json()["total"] == 8


def test_concurrent_duplicate_creates_commit_only_once(tmp_path):
    with TestClient(create_app(database_path=tmp_path / "duplicate-race.sqlite3")) as client:
        def create_duplicate(_index: int):
            return client.post(
                "/documents", json={"id": "shared-id", "text": "one record"}
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(executor.map(create_duplicate, range(12)))

        statuses = [response.status_code for response in responses]
        assert statuses.count(201) == 1
        assert statuses.count(409) == 11
        assert client.get("/health").json()["document_count"] == 1
