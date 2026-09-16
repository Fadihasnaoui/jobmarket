"""Unit tests for the LLM extractor. All calls mocked — zero real network access."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from jobmarket.skills.llm_extractor import MAX_DESCRIPTION_CHARS, extract


def _mock_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 20):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_extract_parses_valid_json() -> None:
    valid = json.dumps(
        {
            "skills": ["Python", "AWS"],
            "seniority": "senior",
            "years_experience_min": 5,
            "is_remote": True,
        }
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(valid, 150, 30)

    result = extract("Senior Data Engineer", "We need Python and AWS experience.", client=client)

    assert result.extraction is not None
    assert result.extraction.skills == ["Python", "AWS"]
    assert result.extraction.seniority == "senior"
    assert result.extraction.years_experience_min == 5
    assert result.extraction.is_remote is True
    assert result.prompt_tokens == 150
    assert result.completion_tokens == 30
    assert result.attempts == 1
    assert client.chat.completions.create.call_count == 1


def test_extract_retries_once_on_malformed_json() -> None:
    client = MagicMock()
    valid = json.dumps(
        {"skills": ["Java"], "seniority": None, "years_experience_min": None, "is_remote": None}
    )
    client.chat.completions.create.side_effect = [
        _mock_response("not json at all"),
        _mock_response(valid),
    ]

    result = extract("Java Developer", "Java required.", client=client)

    assert result.extraction is not None
    assert result.extraction.skills == ["Java"]
    assert result.attempts == 2
    assert client.chat.completions.create.call_count == 2


def test_extract_gives_up_after_max_attempts() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("still not json")

    result = extract("Some Job", "Some description.", client=client)

    assert result.extraction is None
    assert result.error is not None
    assert result.attempts == 2
    assert client.chat.completions.create.call_count == 2


def test_extract_rejects_wrong_shaped_json_and_retries() -> None:
    client = MagicMock()
    bad_shape = json.dumps(
        {"skills": "Python", "seniority": None, "years_experience_min": None, "is_remote": None}
    )
    valid = json.dumps(
        {"skills": ["Python"], "seniority": None, "years_experience_min": None, "is_remote": None}
    )
    client.chat.completions.create.side_effect = [_mock_response(bad_shape), _mock_response(valid)]

    result = extract("Python Dev", "desc", client=client)

    assert result.extraction is not None
    assert result.extraction.skills == ["Python"]
    assert result.attempts == 2


def test_extract_truncates_long_description() -> None:
    client = MagicMock()
    valid = json.dumps(
        {"skills": [], "seniority": None, "years_experience_min": None, "is_remote": None}
    )
    client.chat.completions.create.return_value = _mock_response(valid)

    long_description = "x" * (MAX_DESCRIPTION_CHARS + 500)
    extract("Some Job", long_description, client=client)

    sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "x" * (MAX_DESCRIPTION_CHARS + 1) not in user_content
    assert "x" * MAX_DESCRIPTION_CHARS in user_content


def test_extract_defaults_missing_fields() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(json.dumps({}))

    result = extract("Some Job", "desc", client=client)

    assert result.extraction is not None
    assert result.extraction.skills == []
    assert result.extraction.seniority is None
    assert result.extraction.years_experience_min is None
    assert result.extraction.is_remote is None
