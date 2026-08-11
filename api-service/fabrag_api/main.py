"""FastAPI boundary for the FabRAG question-answering pipeline."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from src.answer import GeneratedAnswer, answer_question

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FabRAG API",
    version="0.1.0",
    description="Grounded answers over electronics-manufacturing documents.",
)


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
    return AnswerResponse(question=result.question, answer=result.text, sources=sources)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/answers", response_model=AnswerResponse)
def create_answer(request: AnswerRequest) -> AnswerResponse:
    try:
        result = answer_question(request.question, request.candidate_k, request.top_n)
    except Exception as exc:
        logger.exception("FabRAG answer pipeline failed", exc_info=exc)
        raise HTTPException(status_code=503, detail="answer pipeline unavailable") from exc
    return _to_response(result)
