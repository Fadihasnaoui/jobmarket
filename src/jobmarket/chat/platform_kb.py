"""Small, fixed knowledge base about this platform's own features.

Deliberately not LLM-generated and not vector-searched: these are short, accurate,
hand-written descriptions of real features (matching what the actual UI/API does
today), matched by keyword. If a feature description ever drifts from the real
implementation, it's a one-line edit here — not a retrieval/grounding problem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformTopic:
    id: str
    keywords: tuple[str, ...]
    answer: str


_TOPICS: tuple[PlatformTopic, ...] = (
    PlatformTopic(
        id="cv_upload",
        keywords=(
            "upload my cv",
            "upload a cv",
            "upload cv",
            "extraction confidence",
            "how is my cv",
            "parse my cv",
            "extract my cv",
            "cv extraction",
        ),
        answer=(
            "You upload a CV (.pdf, .docx, or .txt) on the Upload tab. It's parsed, "
            "then an LLM extracts structured facts (skills, experience, education, "
            "seniority) grounded in exact quotes from your text — nothing is invented. "
            "If the LLM call fails, a deterministic parser is used instead, and this is "
            "always shown honestly (extraction method, confidence score, any degraded "
            "warnings). Nothing is stored beyond a short-lived session."
        ),
    ),
    PlatformTopic(
        id="matching",
        keywords=(
            "job matching",
            "how does matching work",
            "how do matches work",
            "match score",
            "final score",
            "lexical",
            "semantic",
            "hybrid mode",
            "retrieval source",
        ),
        answer=(
            "Job matches are scored by comparing your CV's skills, seniority, domains, "
            "and job type against each posting, in one of three modes: lexical "
            "(keyword/skill overlap), semantic (embedding similarity via the corpus's "
            "pgvector index), or hybrid (a weighted blend of both). Each match shows its "
            "score breakdown (skill, career-level, domain, location components) and "
            "which matched/missing skills drove it."
        ),
    ),
    PlatformTopic(
        id="skill_gap",
        keywords=(
            "skill gap",
            "missing skills",
            "what skills am i missing",
            "recommended skills",
        ),
        answer=(
            "The Skill Gap page aggregates the skills your top job matches ask for that "
            "aren't in your CV, ranked by how many of your matches need them and by "
            "overall demand across the corpus — so it prioritizes genuinely valuable "
            "gaps, not just any missing keyword. A skill you already have under a "
            "recognized domain (e.g. 'AI') is never listed as a gap just because it "
            "isn't a literal line-item skill."
        ),
    ),
    PlatformTopic(
        id="cv_improve_quality",
        keywords=(
            "improve my cv",
            "cv improvement",
            "quality report",
            "cv score",
            "rewrite my cv",
        ),
        answer=(
            "The CV Improve/Quality Report features rework your CV's wording using only "
            "facts already in it (or skills your real top matches call for) — every "
            "generated sentence is checked against your original evidence and any "
            "unsupported claim is rejected, not silently kept. The Quality Report also "
            "scores things like unquantified achievements and spelling."
        ),
    ),
    PlatformTopic(
        id="corpus_data_source",
        keywords=(
            "where does the data come from",
            "data source",
            "how many jobs are in the corpus",
            "how up to date",
            "how fresh",
            "job postings come from",
            "snapshot",
        ),
        answer=(
            "The job corpus is ingested from Adzuna, Jooble, and RemoteOK across six "
            "European countries (France, Germany, Spain, UK, Italy, Netherlands). It's "
            "a point-in-time snapshot, refreshed by re-running ingest — some postings "
            "may have expired on the source site since collection, which is why job "
            "cards show a posting date and flag older listings rather than presenting "
            "everything as equally current."
        ),
    ),
    PlatformTopic(
        id="chat_assistant",
        keywords=(
            "what can you do",
            "what can this chatbot",
            "who are you",
            "what is this assistant",
            "help me understand this chat",
        ),
        answer=(
            "I can answer questions about this platform's real job-market data — the "
            "job corpus, skills demand, salaries, and market trends — plus how the "
            "platform's own features work. Numeric questions are answered from real "
            "database queries, never guessed. I can't help with anything outside this "
            "platform's data (general knowledge, coding help, personal advice, etc.)."
        ),
    ),
)


def match_platform_topic(question: str) -> PlatformTopic | None:
    lowered = question.casefold()
    best: PlatformTopic | None = None
    best_score = 0
    for topic in _TOPICS:
        score = sum(1 for keyword in topic.keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best = topic
    return best


__all__ = ["PlatformTopic", "match_platform_topic"]
