"""Orchestrates one chat turn: classify, then dispatch to the matching answer path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.chat.platform_kb import match_platform_topic
from jobmarket.chat.rag_answerer import answer_rag_question
from jobmarket.chat.router import ChatRoute, classify_question
from jobmarket.chat.sql_answerer import answer_sql_question
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.skills.ontology import SkillsOntology

OFF_TOPIC_MESSAGE = "I can only help with questions about the job market and this platform's data."


@dataclass(frozen=True)
class ChatSource:
    label: str
    url: str | None = None


@dataclass(frozen=True)
class ChatResult:
    route: ChatRoute
    answer: str
    sources: list[ChatSource] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    router_reason: str = ""


async def answer_chat_question(
    session: AsyncSession,
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    ontology: SkillsOntology,
    encoder: EmbeddingEncoder | None = None,
    llm_client: OpenAI | None = None,
    run_id: int | None = None,
) -> ChatResult:
    question = question.strip()
    if not question:
        return ChatResult(
            route=ChatRoute.OFF_TOPIC,
            answer="Ask me something about the job market or this platform's data.",
            router_reason="empty question",
        )

    decision = classify_question(question, history=history, client=llm_client)

    if decision.route == ChatRoute.OFF_TOPIC:
        return ChatResult(
            route=ChatRoute.OFF_TOPIC, answer=OFF_TOPIC_MESSAGE, router_reason=decision.reason
        )

    if decision.route == ChatRoute.PLATFORM:
        topic = match_platform_topic(question)
        answer = topic.answer if topic is not None else OFF_TOPIC_MESSAGE
        return ChatResult(
            route=ChatRoute.PLATFORM,
            answer=answer,
            data={"topic_id": topic.id if topic is not None else None},
            router_reason=decision.reason,
        )

    if decision.route == ChatRoute.SQL:
        sql_answer = await answer_sql_question(session, question, ontology, run_id=run_id)
        data = {
            "intent": sql_answer.intent.value,
            "run_id": sql_answer.run_id,
            **sql_answer.data,
        }
        return ChatResult(
            route=ChatRoute.SQL,
            answer=sql_answer.answer,
            data=data,
            router_reason=decision.reason,
        )

    rag_answer = await answer_rag_question(
        session, question, run_id=run_id, encoder=encoder, client=llm_client, history=history
    )
    sources = [
        ChatSource(label=f"{job.title} at {job.company}", url=job.url) for job in rag_answer.sources
    ]
    return ChatResult(
        route=ChatRoute.RAG,
        answer=rag_answer.answer,
        sources=sources,
        data={"run_id": rag_answer.run_id, "retrieved_count": len(rag_answer.sources)},
        router_reason=decision.reason,
    )


__all__ = ["OFF_TOPIC_MESSAGE", "ChatResult", "ChatSource", "answer_chat_question"]
