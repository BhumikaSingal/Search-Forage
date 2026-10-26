"""FastAPI interface for the persistent SearchForge search service."""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from searchforge.service import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    SearchService,
)
from searchforge.search import SearchEngine

logger = logging.getLogger(__name__)
MAX_PAGE_SIZE = 100
MAX_PAGE_OFFSET = 10_000


class DocumentInput(BaseModel):
    """Document payload accepted by create/update endpoints."""

    id: str = Field(min_length=1, max_length=256)
    text: str = Field(max_length=1_000_000)

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be blank")
        return value


class DocumentOutput(BaseModel):
    id: str
    text: str


class DocumentListOutput(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentOutput]


class SearchHit(BaseModel):
    id: str
    text: str
    score: float


class SearchOutput(BaseModel):
    query: str
    total: int
    limit: int
    offset: int
    results: list[SearchHit]


class HealthOutput(BaseModel):
    status: str
    document_count: int
    vocabulary_size: int


def create_app(
    engine: SearchEngine | None = None,
    database_path: str | Path | None = None,
) -> FastAPI:
    """Create an API application backed by SQLite and a rebuilt search engine."""
    initial_documents = None
    if engine is not None:
        initial_documents = {
            doc_id: text
            for doc_id in engine.document_ids
            if (text := engine.get_document(doc_id)) is not None
        }

    if database_path is None:
        database_path = (
            ":memory:"
            if engine is not None
            else os.environ.get(
                "SEARCHFORGE_DB_PATH",
                str(Path.home() / ".searchforge" / "searchforge.sqlite3"),
            )
        )

    service = SearchService(database_path, initial_documents)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.close()

    app = FastAPI(
        title="SearchForge API",
        description="A SQLite-backed document search service using BM25 ranking.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.search_service = service
    app.state.search_engine = service.search_engine

    @app.exception_handler(RequestValidationError)
    def validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {"location": error["loc"], "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": details,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            status.HTTP_404_NOT_FOUND: "not_found",
            status.HTTP_409_CONFLICT: "conflict",
            status.HTTP_422_UNPROCESSABLE_ENTITY: "validation_error",
        }.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
        )

    @app.exception_handler(DuplicateDocumentError)
    def duplicate_document_error(_: Request, exc: DuplicateDocumentError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "code": "conflict",
                    "message": f"Document {str(exc)!r} already exists",
                }
            },
        )

    @app.exception_handler(DocumentNotFoundError)
    def missing_document_error(_: Request, __: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"code": "not_found", "message": "Document not found"}},
        )

    @app.exception_handler(Exception)
    def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled SearchForge API exception", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred",
                }
            },
        )

    @app.get("/health", response_model=HealthOutput, tags=["system"])
    def health() -> HealthOutput:
        return HealthOutput(
            status="ok",
            document_count=service.document_count,
            vocabulary_size=service.vocabulary_size,
        )

    @app.post(
        "/documents",
        response_model=DocumentOutput,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    def create_document(payload: DocumentInput) -> DocumentOutput:
        document = service.create_document(payload.id, payload.text)
        app.state.search_engine = service.search_engine
        return DocumentOutput(id=document.id, text=document.text)

    @app.get("/documents", response_model=DocumentListOutput, tags=["documents"])
    def list_documents(
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
        offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
    ) -> DocumentListOutput:
        total, page = service.list_documents_page(limit, offset)
        documents = [DocumentOutput(id=document.id, text=document.text) for document in page]
        return DocumentListOutput(
            total=total,
            limit=limit,
            offset=offset,
            documents=documents,
        )

    @app.get("/documents/{doc_id}", response_model=DocumentOutput, tags=["documents"])
    def get_document(doc_id: str) -> DocumentOutput:
        document = service.get_document(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentOutput(id=document.id, text=document.text)

    @app.put("/documents/{doc_id}", response_model=DocumentOutput, tags=["documents"])
    def update_document(doc_id: str, payload: DocumentInput) -> DocumentOutput:
        if doc_id != payload.id:
            raise HTTPException(
                status_code=422,
                detail="Path doc_id must match body id",
            )
        document = service.update_document(doc_id, payload.text)
        app.state.search_engine = service.search_engine
        return DocumentOutput(id=document.id, text=document.text)

    @app.delete(
        "/documents/{doc_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        tags=["documents"],
    )
    def delete_document(doc_id: str) -> Response:
        service.delete_document(doc_id)
        app.state.search_engine = service.search_engine
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/search", response_model=SearchOutput, tags=["search"])
    def search(
        q: str = Query(..., max_length=2_000),
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 10,
        offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
    ) -> SearchOutput:
        ranked = service.search_documents(q)
        page = ranked[offset : offset + limit]
        results = [
            SearchHit(id=document.id, text=document.text, score=score)
            for document, score in page
        ]
        return SearchOutput(
            query=q,
            total=len(ranked),
            limit=limit,
            offset=offset,
            results=results,
        )

    return app


app = create_app()
