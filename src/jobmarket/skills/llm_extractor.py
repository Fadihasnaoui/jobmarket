"""Stage B: LLM-based skill/seniority/experience/remote extraction.

Provider-agnostic via config: currently targets any OpenAI-compatible chat
completions API (Groq, OpenAI itself, or a self-hosted OpenAI-compatible endpoint) —
see `Settings.llm_provider`/`llm_model`/`llm_base_url`/`llm_api_key`. Swapping within
this family (e.g. Groq -> OpenAI) is a config change only. A genuinely different SDK
shape (e.g. native Anthropic) would need a new branch in `_client()`, but the
`extract()` call-site interface would not need to change.

This module is the extraction primitive only — no persistence, no batching, no
job_skills/unmapped_skills writes. That's the enrich runner's job (built once this
primitive is validated against real cost/benefit numbers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from jobmarket.config import get_settings

MAX_DESCRIPTION_CHARS = 3000
MAX_ATTEMPTS = 2  # one retry on malformed JSON

_SYSTEM_PROMPT = (
    "You extract structured information from job postings for a job-market "
    "analytics pipeline. Only report what is explicitly stated in the text — never "
    "guess or infer beyond what's written. The postings are a mix of French and "
    "English. Respond with a single JSON object matching the schema, no prose."
)


def _build_user_prompt(title: str, description: str) -> str:
    return (
        f"Job title: {title}\n\n"
        f"Description:\n{description[:MAX_DESCRIPTION_CHARS]}\n\n"
        "Return a JSON object with exactly these fields:\n"
        '- "skills": array of strings — specific technologies, tools, languages, '
        "frameworks, platforms, or methodologies explicitly named in the text (as "
        "written; do not normalize, translate, or invent ones that aren't there)\n"
        '- "seniority": one of "junior", "mid", "senior", or null if not stated '
        '(French idioms count: "confirmé" ~= mid, "débutant" ~= junior)\n'
        '- "years_experience_min": integer minimum years of experience required, or '
        "null if not stated\n"
        '- "is_remote": true or false ONLY if explicitly stated in the text, '
        "otherwise null — never guess"
    )


class LLMExtraction(BaseModel):
    """Validated shape of one LLM extraction call."""

    skills: list[str] = []
    seniority: str | None = None
    years_experience_min: int | None = None
    is_remote: bool | None = None


@dataclass(frozen=True)
class ExtractionResult:
    extraction: LLMExtraction | None
    prompt_tokens: int
    completion_tokens: int
    raw_response: str
    attempts: int
    error: str | None = None


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def extract(title: str, description: str, *, client: OpenAI | None = None) -> ExtractionResult:
    """One LLM call: skills + seniority + years_experience_min + is_remote.

    Retries once (MAX_ATTEMPTS) if the model's response isn't valid JSON matching
    the schema.
    """
    settings = get_settings()
    llm_client = client or _client()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(title, description)},
    ]

    raw = ""
    prompt_tokens = 0
    completion_tokens = 0
    last_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,  # type: ignore[call-overload]
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        try:
            parsed = LLMExtraction.model_validate(json.loads(raw))
            return ExtractionResult(parsed, prompt_tokens, completion_tokens, raw, attempt)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if attempt < MAX_ATTEMPTS:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON matching the schema. "
                            "Return ONLY the JSON object, no other text."
                        ),
                    }
                )

    return ExtractionResult(None, prompt_tokens, completion_tokens, raw, MAX_ATTEMPTS, last_error)
