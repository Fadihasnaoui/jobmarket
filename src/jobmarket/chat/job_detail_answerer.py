"""Exact, deterministic details for a job selected from the CV Matches page."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.chat.router import ChatRoute
from jobmarket.chat.service import ChatResult, ChatSource
from jobmarket.db.models import Company, Job


def _format_salary(job: Job) -> str:
    if job.salary_min is None and job.salary_max is None:
        return "Not disclosed"
    currency = (job.salary_currency or "currency unspecified").upper()
    period = job.salary_period or "period unspecified"
    if job.salary_min is not None and job.salary_max is not None:
        return f"{currency} {float(job.salary_min):,.0f} - {float(job.salary_max):,.0f} per {period}"
    amount = job.salary_min if job.salary_min is not None else job.salary_max
    return f"{currency} {float(amount):,.0f} per {period}"


async def answer_selected_job_question(session: AsyncSession, job_id: int) -> ChatResult:
    """Return the selected posting's actual stored fields, with no retrieval or LLM."""
    row = (
        await session.execute(
            select(Job, Company.name)
            .select_from(Job)
            .join(Company, Company.id == Job.company_id)
            .where(Job.id == job_id)
        )
    ).one_or_none()

    if row is None:
        return ChatResult(
            route=ChatRoute.OFF_TOPIC,
            answer="That job is no longer available in the current corpus.",
            router_reason="selected job was not found",
        )

    job, company = row
    location = ", ".join(part for part in (job.city, job.country) if part) or job.location_raw or "Not specified"
    remote = "Yes" if job.is_remote else "No"
    posted = job.posted_at.isoformat() if job.posted_at else "Not specified"
    description = job.description.strip() or "No description was supplied with this posting."
    answer = (
        f"{job.title} at {company}\n\n"
        f"Location: {location}\n"
        f"Contract: {job.contract_type or 'Not specified'}\n"
        f"Remote: {remote}\n"
        f"Salary: {_format_salary(job)}\n"
        f"Posted: {posted}\n\n"
        f"Description:\n{description}"
    )
    return ChatResult(
        route=ChatRoute.RAG,
        answer=answer,
        sources=[ChatSource(label=f"{job.title} at {company}", url=job.url)],
        data={"job_id": job.id, "salary": _format_salary(job)},
        router_reason="exact job selected from CV matches",
    )


__all__ = ["answer_selected_job_question"]
