"""Unit tests for deterministic entity extraction — no DB, no LLM."""

from __future__ import annotations

from jobmarket.chat.entities import (
    extract_contract_type,
    extract_country,
    extract_role_title,
    extract_seniority,
    extract_skills,
)
from jobmarket.skills.ontology import load_ontology

ONTOLOGY = load_ontology()


def test_extract_skills_finds_multi_word_skill_over_substrings() -> None:
    skills = extract_skills("what percent of jobs need machine learning and python?", ONTOLOGY)
    assert "Machine Learning" in skills
    assert "Python" in skills


def test_extract_skills_returns_empty_when_nothing_matches() -> None:
    assert extract_skills("what is the weather like today?", ONTOLOGY) == []


def test_extract_country_matches_code_and_full_name() -> None:
    assert extract_country("how many jobs are in France?") == "fr"
    assert extract_country("jobs in the United Kingdom") == "gb"
    assert extract_country("salary in Germany") == "de"
    assert extract_country("no country mentioned here") is None


def test_extract_seniority_matches_known_levels() -> None:
    assert extract_seniority("how many senior roles are there?") == "senior"
    assert extract_seniority("junior data scientist jobs") == "junior"
    assert extract_seniority("no seniority mentioned") is None


def test_extract_contract_type_matches_known_values() -> None:
    assert extract_contract_type("full-time jobs in France") == "full_time"
    assert extract_contract_type("part time roles") == "part_time"
    assert extract_contract_type("freelance opportunities") == "contract"
    assert extract_contract_type("how many jobs are there in France") is None


def test_extract_role_title_matches_known_roles() -> None:
    assert extract_role_title("top skills for Data Scientists?") == "data scientist"
    assert extract_role_title("average salary of Data Engineers?") == "data engineer"
    assert extract_role_title("no role mentioned") is None

def test_extract_country_does_not_treat_ordinary_it_as_italy() -> None:
    assert extract_country("can you explain how it works?") is None
    assert extract_country("what is the average salary in Italy?") == "it"
    assert extract_country("what is the average salary for country=IT?") == "it"
