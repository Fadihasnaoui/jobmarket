"""Router classification tests. Deterministic paths (platform/SQL) never touch an
LLM — verified here by simply not passing a client and confirming they still resolve
correctly. The RAG-vs-off-topic disambiguation does call the LLM; every such test
injects a `MagicMock` client, exactly like `test_cv_llm_extraction.py`'s convention.
Zero real network access anywhere in this file.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from openai import BadRequestError

from jobmarket.chat.router import ChatRoute, classify_question


def _mock_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _mock_client(route: str, reason: str = "test") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        json.dumps({"route": route, "reason": reason})
    )
    return client


def _client_that_fails_if_called() -> MagicMock:
    """A stand-in client whose `.create()` raises -- used to prove a code path
    genuinely never reaches the LLM, not just that it happens to return the right
    route by coincidence."""
    client = MagicMock()
    client.chat.completions.create.side_effect = AssertionError(
        "LLM was called for a question that should have been resolved deterministically"
    )
    return client


def test_analytical_questions_route_to_sql_without_any_llm_call() -> None:
    client = _client_that_fails_if_called()
    for question in [
        "what percent of jobs need Docker?",
        "how many jobs are in France?",
        "top skills for Data Scientists?",
        "average salary of Data Engineers?",
        "how many jobs are there in total?",
    ]:
        decision = classify_question(question, client=client)
        assert decision.route == ChatRoute.SQL, question
    client.chat.completions.create.assert_not_called()


def test_french_analytical_questions_route_to_sql_without_any_llm_call() -> None:
    """Regression test: a French analytical question ("combien d'offres il y a en
    France ?") was previously falling through to the LLM disambiguation stage, which
    only offered rag/off_topic and forced it to off_topic even after correctly
    recognizing it as a statistic request. French analytical phrasing must be caught
    by the deterministic regex, same as English.
    """
    client = _client_that_fails_if_called()
    for question in [
        "combien d'offres il y a en France ?",
        "combien de postes y a-t-il en Allemagne ?",
        "quel pourcentage des offres demandent Docker ?",
        "quel est le salaire moyen des Data Engineers ?",
        "quelles sont les compétences les plus demandées ?",
    ]:
        decision = classify_question(question, client=client)
        assert decision.route == ChatRoute.SQL, question
    client.chat.completions.create.assert_not_called()


def test_platform_questions_route_to_platform_without_any_llm_call() -> None:
    client = _client_that_fails_if_called()
    for question in [
        "how does CV extraction work?",
        "what is the skill gap?",
        "how does job matching work?",
    ]:
        decision = classify_question(question, client=client)
        assert decision.route == ChatRoute.PLATFORM, question
    client.chat.completions.create.assert_not_called()


def test_off_topic_question_classified_off_topic_via_llm() -> None:
    client = _mock_client("off_topic", "general knowledge question")
    decision = classify_question("what is the capital of France?", client=client)
    assert decision.route == ChatRoute.OFF_TOPIC
    client.chat.completions.create.assert_called_once()


def test_exploratory_question_classified_rag_via_llm() -> None:
    client = _mock_client("rag", "asks about real jobs")
    decision = classify_question("tell me about machine learning roles", client=client)
    assert decision.route == ChatRoute.RAG


def test_llm_disambiguation_can_route_to_sql() -> None:
    """Numeric questions that slip past the regex (unanticipated phrasing/language)
    must still be able to reach SQL via the LLM disambiguation stage, per the
    "prefer SQL for anything numeric" instruction in the router prompt -- not be
    forced into rag/off_topic just because those were previously the only options.
    """
    client = _mock_client("sql", "asks for a precise statistic")
    decision = classify_question("wie viele Stellen gibt es in Berlin?", client=client)
    assert decision.route == ChatRoute.SQL


def test_llm_failure_fails_closed_to_off_topic() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network error")
    decision = classify_question("tell me about AI companies", client=client)
    assert decision.route == ChatRoute.OFF_TOPIC


def test_llm_bad_request_retries_then_fails_closed() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = BadRequestError(
        message="bad request", response=MagicMock(), body=None
    )
    decision = classify_question("tell me about AI companies", client=client)
    assert decision.route == ChatRoute.OFF_TOPIC
    assert client.chat.completions.create.call_count == 2  # one retry, per MAX_ATTEMPTS


def test_llm_invalid_json_fails_closed() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("not valid json")
    decision = classify_question("tell me about AI companies", client=client)
    assert decision.route == ChatRoute.OFF_TOPIC


def test_llm_missing_route_field_defaults_off_topic() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(json.dumps({"reason": "no route"}))
    decision = classify_question("tell me about AI companies", client=client)
    assert decision.route == ChatRoute.OFF_TOPIC

def test_feature_how_to_questions_win_over_analytical_keywords() -> None:
    client = _client_that_fails_if_called()
    cases = [
        ("how does job matching work?", "matching"),
        ("comment fonctionne le skill gap ?", "skill_gap"),
        ("explique-moi comment marche le job matching", "matching"),
        ("can you explain how the Skill Gap works on this website?", "skill_gap"),
    ]
    for question, topic_id in cases:
        decision = classify_question(question, client=client)
        assert decision.route == ChatRoute.PLATFORM, question
        assert decision.platform_topic_id == topic_id
        assert "before analytical routing" in decision.reason
    client.chat.completions.create.assert_not_called()


def test_average_salary_questions_remain_sql_data_questions() -> None:
    client = _client_that_fails_if_called()
    for question in [
        "what is the average salary in Italy?",
        "quel est le salaire moyen en Italie ?",
    ]:
        decision = classify_question(question, client=client)
        assert decision.route == ChatRoute.SQL, question
    client.chat.completions.create.assert_not_called()
