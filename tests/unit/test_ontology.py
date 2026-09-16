"""Table-driven tests for the skills ontology loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobmarket.skills.ontology import DEFAULT_ONTOLOGY_PATH, load_ontology, ontology_version


def _write_ontology(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "ontology.yaml"
    path.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
    return path


def test_loads_real_ontology_without_error() -> None:
    ontology = load_ontology(DEFAULT_ONTOLOGY_PATH)
    assert len(ontology) >= 300


@pytest.mark.parametrize(
    ("raw_skill", "expected_canonical"),
    [
        ("Python", "Python"),
        ("python3", "Python"),
        ("PYTHON 3", "Python"),
        ("  python3  ", "Python"),
        ("py", "Python"),
        ("AWS", "AWS"),
        ("amazon web services", "AWS"),
        ("apprentissage automatique", "Machine Learning"),
        ("ML", "Machine Learning"),
        ("devops", "DevOps"),
        ("\u00e9quipe devops", "DevOps"),
        ("k8s", "Kubernetes"),
        ("ci/cd", "CI/CD"),
        ("int\u00e9gration continue", "CI/CD"),
        ("cybers\u00e9curit\u00e9", "Cybersecurity"),
        ("fast-api", "FastAPI"),

        ("KI", "Artificial Intelligence"),
        ("APIs RESTful", "API Design"),
        ("agile Methoden", "Agile"),
        ("Cloud", "Cloud Computing"),
        ("Analytics", "Data Analysis"),
        ("Data Analysis", "Data Analysis"),
        ("Software Engineering", "Software Engineering"),
        ("IT Infrastructure", "IT Infrastructure"),
        ("Agentic Workflows", "Agentic AI"),
        ("Applied AI", "Artificial Intelligence"),
        ("AI-Generated Code", "Generative AI"),
        ("TCP/IP", "TCP/IP"),
        ("tcp ip", "TCP/IP"),
        ("BGP", "BGP"),
        ("SD-WAN", "SD-WAN"),
        ("Palo Alto", "Palo Alto Networks"),
        ("Cisco ASA", "Cisco ASA"),
        ("AWS VPC", "AWS VPC"),
        ("Azure VNet", "Azure Virtual Network"),
        ("nonexistent skill xyz", None),
        ("", None),
    ],
)
def test_resolve_real_ontology(raw_skill: str, expected_canonical: str | None) -> None:
    ontology = load_ontology(DEFAULT_ONTOLOGY_PATH)
    assert ontology.resolve(raw_skill) == expected_canonical


def test_java_does_not_match_inside_javascript() -> None:
    """Guards against alias collisions between short and compound skill names."""
    ontology = load_ontology(DEFAULT_ONTOLOGY_PATH)
    assert ontology.resolve("JavaScript") == "JavaScript"
    assert ontology.resolve("Java") == "Java"


def test_duplicate_canonical_raises(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [
            {"canonical": "Python", "category": "language", "aliases": [], "parent": None},
            {"canonical": "Python", "category": "language", "aliases": [], "parent": None},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate canonical"):
        load_ontology(path)


def test_alias_claimed_by_two_canonicals_raises(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [
            {"canonical": "Python", "category": "language", "aliases": ["py"], "parent": None},
            {"canonical": "PyTorch", "category": "framework", "aliases": ["py"], "parent": None},
        ],
    )
    with pytest.raises(ValueError, match="claimed by both"):
        load_ontology(path)


def test_alias_colliding_with_own_canonical_name_raises(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [
            {"canonical": "Python", "category": "language", "aliases": [], "parent": None},
            {"canonical": "Go", "category": "language", "aliases": ["python"], "parent": None},
        ],
    )
    with pytest.raises(ValueError, match="claimed by both"):
        load_ontology(path)


def test_unknown_category_raises(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [{"canonical": "Python", "category": "not_a_real_category", "aliases": [], "parent": None}],
    )
    with pytest.raises(ValueError, match="Unknown category"):
        load_ontology(path)


def test_unknown_parent_raises(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [
            {
                "canonical": "FastAPI",
                "category": "framework",
                "aliases": [],
                "parent": "Nonexistent",
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown parent"):
        load_ontology(path)


def test_valid_parent_reference_is_fine(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [
            {"canonical": "Python", "category": "language", "aliases": [], "parent": None},
            {"canonical": "FastAPI", "category": "framework", "aliases": [], "parent": "Python"},
        ],
    )
    ontology = load_ontology(path)
    assert ontology.resolve("FastAPI") == "FastAPI"


def test_case_and_whitespace_insensitive_matching(tmp_path: Path) -> None:
    path = _write_ontology(
        tmp_path,
        [
            {
                "canonical": "Kubernetes",
                "category": "tool",
                "aliases": ["k8s", "Kubernetes  Cluster"],
                "parent": None,
            }
        ],
    )
    ontology = load_ontology(path)
    assert ontology.resolve("KUBERNETES") == "Kubernetes"
    assert ontology.resolve("  k8s  ") == "Kubernetes"
    assert ontology.resolve("kubernetes   cluster") == "Kubernetes"


def test_ontology_version_is_semantic_and_formatting_independent(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        """
        # Comments and formatting should not affect the semantic hash.
        - canonical: Python
          category: language
          aliases: [python3, py]
          parent: null
        - canonical: FastAPI
          parent: Python
          aliases:
            - fast api
            - fast-api
          category: framework
        """,
        encoding="utf-8",
    )
    second.write_text(
        """
        - parent: Python
          aliases: [fast-api, fast api]
          category: framework
          canonical: FastAPI

        - aliases:
            - py
            - python3
          canonical: Python
          parent: null
          category: language
        """,
        encoding="utf-8",
    )

    assert ontology_version(load_ontology(first)) == ontology_version(load_ontology(second))


def test_ontology_version_changes_when_semantic_content_changes(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    changed = tmp_path / "changed.yaml"
    base.write_text(
        yaml.safe_dump(
            [
                {
                    "canonical": "Python",
                    "category": "language",
                    "aliases": ["py"],
                    "parent": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    changed.write_text(
        yaml.safe_dump(
            [
                {
                    "canonical": "Python",
                    "category": "language",
                    "aliases": ["py", "python3"],
                    "parent": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    assert ontology_version(load_ontology(base)) != ontology_version(load_ontology(changed))


def test_ontology_version_is_sha256_hex() -> None:
    version = ontology_version(load_ontology(DEFAULT_ONTOLOGY_PATH))

    assert len(version) == 64
    int(version, 16)


def test_gap_cleanup_keeps_excluded_terms_unresolved() -> None:
    ontology = load_ontology(DEFAULT_ONTOLOGY_PATH)

    for raw_skill in [
        "Communication",
        "Problem Solving",
        "Attention to Detail",
        "Leadership",
        "Collaboration",
        "Time Management",
        "English",
        "Customer Service",
        "Accountability",
        "Administrative",
        "Data Entry",
        "Microsoft Office",
        "IT",
        "AI tools",
        "Analytical",
        "analytical skills",
        "Ability to manage multiple deadlines",
    ]:
        assert ontology.resolve(raw_skill) is None


def test_gap_cleanup_version_is_deterministic() -> None:
    first = ontology_version(load_ontology(DEFAULT_ONTOLOGY_PATH))
    second = ontology_version(load_ontology(DEFAULT_ONTOLOGY_PATH))

    assert first == second
