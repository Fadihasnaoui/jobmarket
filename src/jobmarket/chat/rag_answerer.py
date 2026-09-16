"""Exploratory chat answers: vector search over real job postings, LLM answer
strictly grounded in what was retrieved.

Unlike `sql_answerer.py`, this path *does* use the LLM to write the answer — but the
system prompt constrains it to only use the retrieved postings below, explicitly
forbidding outside/world knowledge, and every answer carries its sources (job id,
title, company) so the caller can show what it's grounded in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import BadRequestError, OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.chat.llm_client import build_llm_client
from jobmarket.config import get_settings
from jobmarket.cv.matching import resolve_run_id
from jobmarket.db.models import Company, Job, JobSkill, Skill
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.embeddings.query import embed_query_text
from jobmarket.embeddings.repository import SemanticSearchFilters, search_jobs_by_embedding

DEFAULT_TOP_K = 8
_DESCRIPTION_EXCERPT_CHARS = 400
MAX_ATTEMPTS = 2

_RAG_SYSTEM_PROMPT = (
    "You are a job-market assistant. You must answer the user's question using ONLY "
    "the retrieved job postings listed below -- never use outside/general knowledge, "
    "even if you already know something about the topic. Every claim in your answer "
    "must be traceable to one of these postings; reference postings by their title and "
    "company when you use them (e.g. \"the Data Engineer role at Acme Corp...\"). If "
    "the retrieved postings don't actually contain enough to answer the question, say "
    "so plainly -- do not fill the gap from what you already know. Keep the answer "
    "concise (a few sentences to a short paragraph)."
)


@dataclass(frozen=True)
class RetrievedJob:
    job_id: int
    title: str
    company: str
    country: str | None
    city: str | None
    contract_type: str | None
    is_remote: bool | None
    description_excerpt: str
    skills: list[str]
    url: str | None
    semantic_score: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RetrievedJob] = field(default_factory=list)
    run_id: int | None = None


async def answer_rag_question(
    session: AsyncSession,
    question: str,
    *,
    run_id: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    encoder: EmbeddingEncoder | None = None,
    client: OpenAI | None = None,
    history: list[dict[str, str]] | None = None,
) -> RagAnswer:
    resolved_run_id = await resolve_run_id(session, run_id)
    query_embedding = embed_query_text(question, encoder=encoder)
    hits = await search_jobs_by_embedding(
        session, query_embedding, limit=top_k, filters=SemanticSearchFilters()
    )
    if not hits:
        return RagAnswer(
            answer=(
                "I couldn't find any relevant jobs in the corpus for that question. "
                "The data may not cover this specific case."
            ),
            sources=[],
            run_id=resolved_run_id,
        )

    retrieved = await _fetch_job_details(
        session, resolved_run_id, {hit.job_id: hit.semantic_score for hit in hits}
    )
    if not retrieved:
        return RagAnswer(
            answer=(
                "I found some semantically similar postings, but couldn't load their "
                "details from the current data run. Try rephrasing or asking again."
            ),
            sources=[],
            run_id=resolved_run_id,
        )

    answer_text = _generate_grounded_answer(
        question, retrieved, history=history, client=client
    )
    return RagAnswer(answer=answer_text, sources=retrieved, run_id=resolved_run_id)


async def _fetch_job_details(
    session: AsyncSession, run_id: int, scores_by_job_id: dict[int, float]
) -> list[RetrievedJob]:
    """Fetch full details for jobs `search_jobs_by_embedding` already found relevant.

    Deliberately does NOT require the job to be in `run_id`'s successfully-enriched
    set: `search_jobs_by_embedding` searches over every embedded job in the corpus,
    a materially broader population than any one enrichment run's coverage (a run can
    cover a subset of jobs -- see `cv/matching.py::resolve_run_id`'s own note on this).
    Gating on enrichment-run success here silently dropped valid semantic hits whose
    job simply wasn't part of that particular run, confirmed live: real questions
    returned zero retrieved jobs despite `search_jobs_by_embedding` finding real hits.
    Skills are still scoped to `run_id` (outer join -- a job outside the run's
    coverage just shows no skills, rather than being excluded entirely).
    """
    job_ids = list(scores_by_job_id)
    rows = (
        await session.execute(
            select(
                Job.id,
                Job.title,
                Company.name,
                Job.country,
                Job.city,
                Job.contract_type,
                Job.is_remote,
                Job.description,
                Job.url,
                Skill.canonical,
            )
            .select_from(Job)
            .join(Company, Company.id == Job.company_id)
            .outerjoin(JobSkill, (JobSkill.job_id == Job.id) & (JobSkill.run_id == run_id))
            .outerjoin(Skill, Skill.id == JobSkill.skill_id)
            .where(Job.id.in_(job_ids))
            .order_by(Job.id, Skill.canonical)
        )
    ).all()

    by_job: dict[int, dict[str, Any]] = {}
    for (
        job_id,
        title,
        company,
        country,
        city,
        contract_type,
        is_remote,
        description,
        url,
        canonical,
    ) in rows:
        entry = by_job.setdefault(
            job_id,
            {
                "title": title,
                "company": company,
                "country": country,
                "city": city,
                "contract_type": contract_type,
                "is_remote": is_remote,
                "description": description,
                "url": url,
                "skills": [],
            },
        )
        if canonical is not None and canonical not in entry["skills"]:
            entry["skills"].append(canonical)

    ordered_ids = sorted(by_job, key=lambda jid: -scores_by_job_id.get(jid, 0.0))
    return [
        RetrievedJob(
            job_id=jid,
            title=by_job[jid]["title"],
            company=by_job[jid]["company"],
            country=by_job[jid]["country"],
            city=by_job[jid]["city"],
            contract_type=by_job[jid]["contract_type"],
            is_remote=by_job[jid]["is_remote"],
            description_excerpt=(by_job[jid]["description"] or "")[:_DESCRIPTION_EXCERPT_CHARS],
            skills=by_job[jid]["skills"],
            url=by_job[jid]["url"],
            semantic_score=round(scores_by_job_id.get(jid, 0.0), 4),
        )
        for jid in ordered_ids
    ]


def _build_context(retrieved: list[RetrievedJob]) -> str:
    lines = []
    for i, job in enumerate(retrieved, start=1):
        location = ", ".join(part for part in (job.city, job.country) if part) or "location unknown"
        skills = ", ".join(job.skills) if job.skills else "no listed skills"
        lines.append(
            f"[{i}] \"{job.title}\" at {job.company} ({location}"
            f"{', ' + job.contract_type if job.contract_type else ''}"
            f"{', remote' if job.is_remote else ''})\n"
            f"    Skills: {skills}\n"
            f"    Excerpt: {job.description_excerpt}"
        )
    return "\n".join(lines)


def _generate_grounded_answer(
    question: str,
    retrieved: list[RetrievedJob],
    *,
    history: list[dict[str, str]] | None,
    client: OpenAI | None,
) -> str:
    settings = get_settings()
    try:
        llm_client = client or build_llm_client()
    except Exception:
        return _fallback_answer(retrieved)

    context = _build_context(retrieved)
    messages: list[dict[str, str]] = [{"role": "system", "content": _RAG_SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"Retrieved job postings:\n{context}\n\nQuestion: {question}",
        }
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
                max_completion_tokens=500,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except BadRequestError:
            if attempt < MAX_ATTEMPTS:
                continue
            return _fallback_answer(retrieved)
        except Exception:
            return _fallback_answer(retrieved)
    return _fallback_answer(retrieved)


def _fallback_answer(retrieved: list[RetrievedJob]) -> str:
    titles = "; ".join(f'"{job.title}" at {job.company}' for job in retrieved[:5])
    return (
        "I found relevant postings but couldn't generate a summary right now. "
        f"The closest matches are: {titles}."
    )


__all__ = ["DEFAULT_TOP_K", "RagAnswer", "RetrievedJob", "answer_rag_question"]
