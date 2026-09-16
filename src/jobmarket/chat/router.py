"""Classifies a chat question into one of four answer paths.

Plain RAG over job postings gives fabricated numbers for analytical questions (it
retrieves a handful of jobs and the LLM guesses a statistic from them). This router
exists specifically to keep numeric/analytical questions away from free-form LLM
generation entirely — see `sql_answerer.py`'s module docstring.

Classification is layered, cheapest/most-certain first:
1. Platform-topic keyword match (`platform_kb.match_platform_topic`) -> PLATFORM.
2. Analytical-intent regex match -> SQL. Deliberately pattern-only, not entity-gated:
   "how many jobs are there" is a valid SQL question with no skill/country entity at
   all; `sql_answerer.py` handles the no-filter case itself.
3. Otherwise, an LLM call decides SQL (a numeric/statistical question phrased in a
   way the regex didn't anticipate, including other languages) vs RAG (on-topic,
   needs real job postings to answer) vs OFF_TOPIC (outside this platform's data
   entirely) — the one classification a fixed keyword list genuinely can't do
   reliably ("tell me about machine learning jobs" vs "explain machine learning"
   need real judgment). The prompt is explicit: prefer SQL whenever a precise
   number is being asked for, and only classify OFF_TOPIC when the question is
   genuinely unrelated to this platform's job-market data.
4. If the LLM call itself fails (network, auth, malformed response), fail CLOSED to
   OFF_TOPIC with a clear reason — never silently fall through to answering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

from jobmarket.chat.platform_kb import match_platform_topic
from jobmarket.config import get_settings

_ANALYTICAL_PATTERNS = (
    r"%|percent|percentage|proportion|fraction|share of|"
    r"pourcentage|proportion de|part de",
    r"\bhow many\b|\bnumber of\b|\bcount of\b|\btotal (number|count)\b|"
    r"\bcombien\b|\bnombre de\b",
    r"\baverage\b|\bmean\b|\btypical salary\b|salary (of|for|in)|"
    r"how much (does|do|is|are|can)|"
    r"\bmoyenne\b|salaire moyen|en moyenne|salaire (de|pour|en)",
    r"\btop \d*\s*skills\b|\bmost common\b|\bmost in-demand\b|\bmost requested\b|"
    r"\bmost popular\b|\bmost needed\b|"
    r"plus demand[ée]s?|plus courants?|plus recherch[ée]s?|plus populaires?",
    r"\bdistribution\b|\bbreakdown\b|\bsplit by\b|\bcompare\b|r[ée]partition\b|comparer\b",
)
_ANALYTICAL_RE = re.compile("|".join(_ANALYTICAL_PATTERNS), re.IGNORECASE)

# The form of an explicit feature how-to question is checked before analytical
# keywords. match_platform_topic identifies the feature itself before the
# analytical keyword patterns have a chance to classify the question as SQL.
_FEATURE_HOW_TO_RE = re.compile(
    r"\bhow\s+(?:does|do|did|can)\b.*\bwork(?:s|ed)?\b|"
    r"\bexplain(?:\s+to\s+me)?\s+how\b.*\bwork(?:s|ed)?\b|"
    r"\bexplain\b.*\bhow\b.*\bwork(?:s|ed)?\b|"
    r"\bcomment\s+(?:ca\s+)?fonctionne\b|"
    r"\bexplique(?:z)?(?:[-\s]?moi)?\s+comment\b.*\bmarche\b",
    re.IGNORECASE,
)

_ROUTER_SYSTEM_PROMPT = (
    "You are a strict topic classifier for a job-market platform's assistant. The "
    "platform's real data covers ONLY: job postings (titles, companies, countries, "
    "skills, salaries, contract types) from its own corpus, skills demand and "
    "job-market trends derived from that corpus, and this platform's own features. "
    "It has no other knowledge and must not use any. Questions may be asked in "
    "English, French, or other languages -- classify by meaning, not language.\n\n"
    "Classify the user question into exactly one of:\n"
    '- "sql": asks for a precise count, percentage, average, or ranked statistic '
    "computed from the job corpus -- e.g. \"how many jobs are in Germany\", "
    "\"combien d'offres il y a en France\", \"quel pourcentage de postes "
    'demandent Docker", "what is the average salary".\n'
    '- "rag": on-topic and about real jobs/companies/roles/skills in a way that '
    "needs looking at actual job postings to answer well (not a precise statistic) "
    "-- e.g. \"tell me about machine learning roles\", \"what companies are hiring "
    'for AI", "what is a Data Engineer job like".\n'
    '- "off_topic": NOT about this platform\'s job-market data -- general knowledge, '
    "unrelated coding help, personal advice, current events, or any other subject "
    "-- e.g. \"what is the capital of France\", \"write me a poem\", \"how do I "
    'center a div".\n\n'
    'Respond with ONLY one strict JSON object: {"route": "sql", "rag", or '
    '"off_topic", "reason": "one short sentence"}. No prose, no markdown.\n\n'
    'If the question asks for a specific number/statistic, prefer "sql" even if '
    "unsure exactly which one -- never guess a number yourself, the SQL path will "
    'report honestly if the data cannot answer it precisely. Only classify '
    '"off_topic" when the question is genuinely unrelated to this platform\'s '
    "job-market data -- wrongly declining a truly off-topic question is a far "
    "smaller failure than answering from outside this platform's real data."
)

MAX_ATTEMPTS = 2


class ChatRoute(StrEnum):
    SQL = "sql"
    RAG = "rag"
    PLATFORM = "platform"
    OFF_TOPIC = "off_topic"


@dataclass(frozen=True)
class RouterDecision:
    route: ChatRoute
    reason: str
    platform_topic_id: str | None = None


class _IntentClassification(BaseModel):
    model_config = {"extra": "ignore"}
    route: str = "off_topic"
    reason: str = ""


def classify_question(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    client: OpenAI | None = None,
) -> RouterDecision:
    """Classify one question into a `ChatRoute`. `history` is a short list of
    `{"role": "user"|"assistant", "content": "..."}` turns, used only to give the
    LLM disambiguation stage conversational context (e.g. "what about in Germany?"
    following an on-topic exchange) -- the deterministic SQL/platform stages look
    only at the current question's own text.
    """
    topic = match_platform_topic(question)
    if topic is not None and _FEATURE_HOW_TO_RE.search(question):
        return RouterDecision(
            route=ChatRoute.PLATFORM,
            reason=f"matched platform how-to topic before analytical routing: {topic.id}",
            platform_topic_id=topic.id,
        )

    if topic is not None:
        return RouterDecision(
            route=ChatRoute.PLATFORM,
            reason=f"matched platform topic: {topic.id}",
            platform_topic_id=topic.id,
        )

    if _ANALYTICAL_RE.search(question):
        return RouterDecision(route=ChatRoute.SQL, reason="matched an analytical/numeric pattern")

    return _classify_on_topic_or_not(question, history=history, client=client)


def _classify_on_topic_or_not(
    question: str,
    *,
    history: list[dict[str, str]] | None,
    client: OpenAI | None,
) -> RouterDecision:
    settings = get_settings()
    try:
        llm_client = client or _build_client()
    except Exception as exc:  # missing API key etc — fail closed, don't crash the request
        return RouterDecision(
            route=ChatRoute.OFF_TOPIC, reason=f"router LLM unavailable, failing closed: {exc}"
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": _ROUTER_SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[call-overload]
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=200,
            )
        except BadRequestError as exc:
            last_error = f"bad_request: {exc}"
            if attempt < MAX_ATTEMPTS:
                continue
            return RouterDecision(route=ChatRoute.OFF_TOPIC, reason=f"failing closed: {last_error}")
        except Exception as exc:  # network/timeout/auth — fail closed immediately
            return RouterDecision(
                route=ChatRoute.OFF_TOPIC, reason=f"router LLM call failed, failing closed: {exc}"
            )
        raw = response.choices[0].message.content or ""
        try:
            parsed = _IntentClassification.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"invalid JSON: {exc}"
            if attempt < MAX_ATTEMPTS:
                continue
            return RouterDecision(route=ChatRoute.OFF_TOPIC, reason=f"failing closed: {last_error}")
        if parsed.route == "sql":
            route = ChatRoute.SQL
        elif parsed.route == "rag":
            route = ChatRoute.RAG
        else:
            route = ChatRoute.OFF_TOPIC
        return RouterDecision(route=route, reason=parsed.reason or "llm classification")

    return RouterDecision(route=ChatRoute.OFF_TOPIC, reason=f"failing closed: {last_error}")


def _build_client() -> OpenAI:
    from jobmarket.chat.llm_client import build_llm_client

    return build_llm_client()


__all__ = ["ChatRoute", "RouterDecision", "classify_question"]
