"""Job-market-restricted chat endpoint — router + SQL/RAG/platform-KB answer paths.

See `chat/router.py`, `chat/sql_answerer.py`, `chat/rag_answerer.py`,
`chat/platform_kb.py`, and `chat/service.py` for the actual logic; this route is a
thin adapter converting `ChatResult` into the API's response shape.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.api.dependencies import (
    get_chat_llm_client,
    get_db_session,
    get_ontology,
    get_upload_encoder,
)
from jobmarket.api.models import ChatRequest, ChatResponse, ChatSourceOut
from jobmarket.chat.job_detail_answerer import answer_selected_job_question
from jobmarket.chat.service import answer_chat_question
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.skills.ontology import SkillsOntology

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest = Body(...),
    session: AsyncSession = Depends(get_db_session),
    ontology: SkillsOntology = Depends(get_ontology),
    encoder: EmbeddingEncoder | None = Depends(get_upload_encoder),
    llm_client: OpenAI | None = Depends(get_chat_llm_client),
) -> ChatResponse:
    history = [{"role": msg.role, "content": msg.content} for msg in body.history]
    if body.job_id is not None:
        result = await answer_selected_job_question(session, body.job_id)
    else:
        result = await answer_chat_question(
            session,
            body.question,
            history=history,
            ontology=ontology,
            encoder=encoder,
            llm_client=llm_client,
            run_id=body.run_id,
        )
    return ChatResponse(
        route=result.route.value,
        answer=result.answer,
        sources=[ChatSourceOut(label=s.label, url=s.url) for s in result.sources],
        data=result.data,
        router_reason=result.router_reason,
    )


__all__ = ["router"]
