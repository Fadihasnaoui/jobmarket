"""Table-driven tests for the deterministic skills matcher."""

from __future__ import annotations

import pytest

from jobmarket.skills.matcher import (
    MATCHER_VERSION,
    find_skills,
    get_matcher_versions,
    match_skill_canonicals,
)
from jobmarket.skills.ontology import SkillEntry, SkillsOntology, load_ontology


@pytest.fixture
def tiny_ontology() -> SkillsOntology:
    return SkillsOntology(
        [
            SkillEntry(canonical="Java", category="language"),
            SkillEntry(canonical="JavaScript", category="language", aliases=["js"]),
            SkillEntry(canonical="R", category="language"),
            SkillEntry(canonical="C", category="language", aliases=["langage c"]),
            SkillEntry(
                canonical="Artificial Intelligence",
                category="domain",
                aliases=["ai", "ia", "intelligence artificielle"],
            ),
            SkillEntry(canonical="CI/CD", category="methodology", aliases=["ci", "cd", "ci/cd"]),
            SkillEntry(canonical="Go", category="language", aliases=["golang"]),
            SkillEntry(canonical="Python", category="language", aliases=["python3"]),
            SkillEntry(canonical="Kubernetes", category="tool", aliases=["k8s"]),
            SkillEntry(
                canonical="Cybersecurity", category="domain", aliases=["cybers\u00e9curit\u00e9"]
            ),
        ]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We use Java and Python.", {"Java", "Python"}),
        ("We use JavaScript for the frontend.", {"JavaScript"}),
        ("JavaScript developers welcome", {"JavaScript"}),
        ("Java developers welcome", {"Java"}),
    ],
)
def test_java_never_matches_inside_javascript(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python, R, SQL required.", {"Python", "R"}),
        ("Comp\u00e9tences: R et Python.", {"R", "Python"}),
        ("(R) is a plus", set()),
        ("CI/CD, R/Python", {"CI/CD", "R", "Python"}),
        ("Notre service R&D innove sans cesse.", set()),
        ("Le budget R&D augmente.", set()),
        ("Filiale P&L en croissance", set()),
    ],
)
def test_r_language_boundary_guards(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C'est une entreprise innovante.", set()),
        ("Nous d\u00e9veloppons en C et en Python.", {"C", "Python"}),
        ("Langage C requis.", {"C"}),
    ],
)
def test_c_language_not_matched_inside_cest(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("J'ai d\u00e9velopp\u00e9 des modÃƒÂ¨les d'IA.", set()),
        ("AI Engineer position open.", {"Artificial Intelligence"}),
        ("Poste en IA et machine learning.", {"Artificial Intelligence"}),
        ("Nous recherchons un expert en intelligence artificielle.", {"Artificial Intelligence"}),
    ],
)
def test_ai_not_matched_inside_jai(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Voir ci-dessous pour plus de d\u00e9tails.", set()),
        ("Voir ci-joint le dossier complet.", set()),
        ("MaÃƒÂ®trise de CI/CD indispensable.", {"CI/CD"}),
        ("Pipeline CI en place depuis 2020.", {"CI/CD"}),
    ],
)
def test_ci_not_matched_inside_ci_dessous(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Exp\u00e9rience en cybers\u00e9curit\u00e9 requise.", {"Cybersecurity"}),
        ("Experience en cybersecurite requise.", {"Cybersecurity"}),
        ("CYBERS\u00c9CURIT\u00c9 avanc\u00e9e.", {"Cybersecurity"}),
    ],
)
def test_accent_insensitive_matching(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python, Go, Rust required.", {"Python", "Go"}),
        ("Let's go build something amazing", set()),
        ("go-getter attitude wanted", set()),
    ],
)
def test_go_language_lowercase_and_neighbor_guards(
    tiny_ontology: SkillsOntology, text: str, expected: set[str]
) -> None:
    assert match_skill_canonicals(text, tiny_ontology) == expected


def test_single_character_languages_require_technical_context(
    tiny_ontology: SkillsOntology,
) -> None:
    assert "R" not in match_skill_canonicals(
        "Bachelor in Management. R Dupont address.", tiny_ontology
    )
    assert "C" not in match_skill_canonicals(
        "Customer service and communication C level.", tiny_ontology
    )
    assert "R" in match_skill_canonicals("Skills: Python, R, SQL", tiny_ontology)
    assert "R" in match_skill_canonicals("Statistical analysis using R and RStudio", tiny_ontology)
    assert "C" in match_skill_canonicals("Programming language C and C/C++", tiny_ontology)


def test_k8s_alias_matches_kubernetes(tiny_ontology: SkillsOntology) -> None:
    assert match_skill_canonicals("Experience with k8s clusters.", tiny_ontology) == {"Kubernetes"}


def test_case_insensitive_for_long_aliases(tiny_ontology: SkillsOntology) -> None:
    assert match_skill_canonicals("PYTHON3 required", tiny_ontology) == {"Python"}
    assert match_skill_canonicals("python3 required", tiny_ontology) == {"Python"}


def test_empty_text_returns_nothing(tiny_ontology: SkillsOntology) -> None:
    assert find_skills("", tiny_ontology) == []


def test_multiple_skills_returned_sorted_by_position(tiny_ontology: SkillsOntology) -> None:
    matches = find_skills("Python puis Java puis Kubernetes", tiny_ontology)
    canonicals_in_order = [m.canonical for m in matches]
    assert canonicals_in_order == ["Python", "Java", "Kubernetes"]


def test_no_false_positive_on_unrelated_text(tiny_ontology: SkillsOntology) -> None:
    text = "Nous recherchons une personne dynamique et autonome pour rejoindre l'\u00e9quipe."
    assert match_skill_canonicals(text, tiny_ontology) == set()



def test_match_offsets_remain_original_after_accent_folding() -> None:
    ontology = SkillsOntology(
        [
            SkillEntry(
                canonical="Data Analysis",
                category="domain",
                aliases=["analytics"],
            )
        ]
    )
    text = "Strat\u00e9gie Data & Analytics Engineering"

    match = find_skills(text, ontology)[0]

    assert match.canonical == "Data Analysis"
    assert text[match.start : match.end] == "Analytics"


def test_match_offsets_remain_original_after_casefold_expansion() -> None:
    ontology = SkillsOntology([SkillEntry(canonical="Python", category="language")])
    text = "Stra\u00dfe team uses Python"

    match = find_skills(text, ontology)[0]

    assert match.canonical == "Python"
    assert text[match.start : match.end] == "Python"


def test_get_matcher_versions_returns_explicit_matcher_and_ontology_versions(
    tiny_ontology: SkillsOntology,
) -> None:
    versions = get_matcher_versions(tiny_ontology)

    assert versions.matcher_version == MATCHER_VERSION
    assert versions.ontology_version == tiny_ontology.version()
    assert len(versions.ontology_version) == 64


def test_ki_alias_requires_uppercase_and_ai_context() -> None:
    ontology = load_ontology()

    assert match_skill_canonicals("Wir bauen KI Modelle", ontology) == {"Artificial Intelligence"}
    assert "Artificial Intelligence" not in match_skill_canonicals(
        "kind regards from berlin", ontology
    )
    assert "Artificial Intelligence" not in match_skill_canonicals(
        "ki ohne fachlichen kontext", ontology
    )


def test_gap_cleanup_aliases_match_without_soft_skill_noise() -> None:
    ontology = load_ontology()

    assert match_skill_canonicals("APIs RESTful et cloud-native", ontology) == {
        "API Design",
        "Cloud Computing",
    }
    assert "Communication" not in match_skill_canonicals("excellent communication", ontology)
