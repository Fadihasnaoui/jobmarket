"""Regression tests for matcher quality corrections from run-75 audit."""

from __future__ import annotations

import csv
from pathlib import Path

from jobmarket.skills.matcher import MATCHER_VERSION, find_skills, match_skill_canonicals
from jobmarket.skills.ontology import load_ontology

FIXTURE = Path("tests/fixtures/matcher_quality_eval.csv")
PREVIOUS_ONTOLOGY_VERSION = "bb7cc3a01768e485fb222086722bfd3b582db5336f7647a1873973fa15289a13"


def test_quality_eval_fixture_matches_expected_and_blocks_negatives() -> None:
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8", newline="")))
    assert len(rows) >= 20

    for row in rows:
        found = match_skill_canonicals(row["text"])
        expected = _split(row["expected_canonicals"])
        blocked = _split(row["must_not_match"])
        assert expected <= found, row
        assert found.isdisjoint(blocked), row


def test_genuine_and_unrelated_ai_ia_cases() -> None:
    assert match_skill_canonicals("AI Engineer position open.") == {"Artificial Intelligence"}
    assert {"Artificial Intelligence", "Machine Learning"} <= match_skill_canonicals(
        "Poste en IA et machine learning."
    )
    assert "Artificial Intelligence" not in match_skill_canonicals("J'ai d?velopp? l'IA.")
    assert "Artificial Intelligence" not in match_skill_canonicals("plain ai lowercase text")


def test_chef_tool_matches_but_french_managerial_contexts_do_not() -> None:
    assert match_skill_canonicals("Automate infrastructure with Chef recipes.") == {"Chef"}
    assert "Chef" not in match_skill_canonicals("Chef de Projet Data Scientist")
    assert "Chef" not in match_skill_canonicals("sous l'autorit? du chef de projet")
    assert "Chef" not in match_skill_canonicals("chef d'equipe data")


def test_ambiguous_short_alias_policy() -> None:
    assert {"Machine Learning"} <= match_skill_canonicals("Senior Data Scientist / ML Engineer")
    assert "Business Intelligence" in match_skill_canonicals("Lead Data Scientist - BI Prediction")
    assert {"C", "C++"} <= match_skill_canonicals("Core product in C/C++ and Python")
    assert "SAP" in match_skill_canonicals("Consultant SAP BW data platform")
    assert "GCP" in match_skill_canonicals("Cloud GCP / Vertex AI platform")
    assert "Retrieval-Augmented Generation" in match_skill_canonicals("LLM, RAG and AI")
    assert "Large Language Models" in match_skill_canonicals("LLM pour mod?les de langage")

    assert "Machine Learning" not in match_skill_canonicals("calme et autonome")
    assert "Business Intelligence" not in match_skill_canonicals("bienveillance attendue")
    assert "C" not in match_skill_canonicals("C'est important")


def test_role_title_ontology_decisions_and_offsets() -> None:
    matches = find_skills("Data Scientist - Freelance")
    assert [(match.canonical, match.start, match.end) for match in matches] == [
        ("Data Science", 0, 14)
    ]
    assert match_skill_canonicals("Machine Learning Engineer") == {"Machine Learning"}
    assert match_skill_canonicals("AI Engineer F/H") == {"Artificial Intelligence"}


def test_matcher_version_incremented_for_offset_mapping_and_ontology_hash_changed() -> None:
    assert MATCHER_VERSION == "deterministic-alias-matcher-v3"
    assert load_ontology().version() != PREVIOUS_ONTOLOGY_VERSION


def _split(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}
