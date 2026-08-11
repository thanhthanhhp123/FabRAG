"""FastAPI boundary for the FabRAG question-answering pipeline."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator
from src.answer import GeneratedAnswer, answer_question

from .observability import (
    REQUEST_ID_HEADER,
    REQUEST_ID_PATTERN,
    configure_access_logger,
    request_id_context,
)
from .security import require_api_key

logger = logging.getLogger(__name__)
access_logger = configure_access_logger()

app = FastAPI(
    title="FabRAG API",
    version="0.1.0",
    description="Grounded answers over electronics-manufacturing documents.",
)


@app.middleware("http")
async def request_context(request: Request, call_next) -> Response:
    supplied_id = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = supplied_id if REQUEST_ID_PATTERN.fullmatch(supplied_id) else uuid.uuid4().hex
    token = request_id_context.set(request_id)
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        access_logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        request_id_context.reset(token)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    candidate_k: int = Field(default=10, ge=1, le=50)
    top_n: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def validate_ranking_limits(self) -> AnswerRequest:
        if self.top_n > self.candidate_k:
            raise ValueError("top_n must not exceed candidate_k")
        if not self.question.strip():
            raise ValueError("question must not be blank")
        self.question = self.question.strip()
        return self


class SourceResponse(BaseModel):
    source_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    reranker_score: float
    retrieval_rank: int
    vector_rank: int | None
    keyword_rank: int | None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    route: str
    sources: list[SourceResponse]


def _to_response(result: GeneratedAnswer) -> AnswerResponse:
    sources = [
        SourceResponse(
            source_id=f"S{index}",
            filename=source.filename,
            page_start=source.page_start,
            page_end=source.page_end,
            chunk_index=source.chunk_index,
            reranker_score=source.reranker_score,
            retrieval_rank=source.retrieval_rank,
            vector_rank=source.vector_rank,
            keyword_rank=source.keyword_rank,
        )
        for index, source in enumerate(result.sources, start=1)
    ]
    return AnswerResponse(
        question=result.question,
        answer=result.text,
        route=result.route.value,
        sources=sources,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/answers", response_model=AnswerResponse)
def create_answer(
    request: AnswerRequest,
    _identity: str = Depends(require_api_key),
) -> AnswerResponse:
    try:
        result = answer_question(request.question, request.candidate_k, request.top_n)
    except Exception as exc:
        logger.exception("FabRAG answer pipeline failed", exc_info=exc)
        raise HTTPException(status_code=503, detail="answer pipeline unavailable") from exc
    return _to_response(result)
