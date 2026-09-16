"""Regression tests for CV management-profile quality and abstention."""

from __future__ import annotations

from datetime import date

from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.profile import DocumentPage, ParsedDocument
from jobmarket.cv.quality import calculate_duration_months

MANAGEMENT_CV = """
Profil
Jeune diplômée Licence en Management intéressée par la transformation digitale,
l'innovation, la communication, le leadership et le travail d'équipe.

Formation
Licence en Management - Université Exemple - diplômée 2025

Expérience professionnelle
06/2024 – 08/2024 Assistante de vente - parfumerie
Accueil client, conseil client, vente de produits, gestion des stocks et encaissement.

01-06-2025 / 31-07-2025 Stage assurance - Cabinet Assurance
Suivi des dossiers de souscription et contrôle des dossiers administratifs.

Jul 2025 – Aug 2025 Commercial support - Education First
Accompagnement des étudiants, soutien commercial et relation client.

Compétences
Communication, leadership, organisation, travail d'équipe, transformation digitale.
"""

LOW_EVIDENCE_CV = """
Profil
R
"""


def test_duration_calculation_supports_full_english_and_french_month_names() -> None:
    assert calculate_duration_months("July 2015", "January 2018") == 31
    assert calculate_duration_months("juillet 2015", "janvier 2018") == 31
    assert calculate_duration_months("January 2018", "Present", as_of=date(2026, 8, 31)) == 104
    assert calculate_duration_months("janvier 2018", "actuel", as_of=date(2026, 8, 31)) == 104
    assert calculate_duration_months("janvier 2018", "aujourd'hui", as_of=date(2026, 8, 31)) == 104


def test_management_cv_does_not_extract_false_r_and_extracts_business_profile() -> None:
    profile = extract_cv_profile(_document(MANAGEMENT_CV))

    assert "R" not in {skill.canonical_skill for skill in profile.skills}
    assert profile.education_detail is not None
    assert profile.education_detail.education_status == "graduated"
    assert profile.education_detail.education_level == "bachelor/licence"
    assert profile.education_detail.education_field == "management"
    assert profile.candidate.career_level == "junior"
    assert profile.candidate.experience.internship_count >= 1
    assert profile.candidate.experience.professional_experience_count >= 2
    assert len(profile.experience_entries) >= 3
    domains = set(profile.candidate.preferred_domains)
    assert {"Management", "Customer Service", "Sales", "Insurance", "Communication"} <= domains
    assert profile.extraction_quality.extraction_confidence_label in {"medium", "high"}
    assert profile.extraction_quality.abstention_reason is None


def test_low_evidence_cv_abstains_and_rejects_ambiguous_r() -> None:
    profile = extract_cv_profile(_document(LOW_EVIDENCE_CV))

    assert profile.skills == []
    assert profile.extraction_quality.extraction_confidence_label == "low"
    assert profile.extraction_quality.abstention_reason == "insufficient_profile_evidence"
    assert "R" in profile.extraction_quality.rejected_ambiguous_aliases


def test_cv_section_detection_multilingual() -> None:
    profile = extract_cv_profile(_document(MANAGEMENT_CV))
    labels = {section.normalized_label for section in profile.sections}

    assert {"profile", "education", "experience", "skills"} <= labels


def _document(text: str) -> ParsedDocument:
    cleaned = text.strip()
    return ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text=cleaned,
        page_count=None,
        pages=[DocumentPage(page=None, text=cleaned, start_offset=0, end_offset=len(cleaned))],
        warnings=[],
        file_hash="a" * 64,
        file_size=len(cleaned.encode()),
    )


QUALITY_FIXTURES: tuple[tuple[str, str, set[str], str | None], ...] = (
    (
        "management_business",
        MANAGEMENT_CV,
        {"Management", "Customer Service", "Sales"},
        None,
    ),
    (
        "data_student",
        "Formation\nMaster data science en cours 2026\nCompétences\nPython SQL Machine Learning",
        {"Data Science"},
        "insufficient_profile_evidence",
    ),
    (
        "software_intern",
        "Experience\nJan 2025 - Jun 2025 Software engineering internship\nSkills\nJava Kubernetes",
        set(),
        "insufficient_profile_evidence",
    ),
    (
        "marketing_profile",
        (
            "Profile\nMarketing assistant with communication, organisation "
            "and public speaking experience."
        ),
        {"Marketing", "Communication"},
        "insufficient_profile_evidence",
    ),
    (
        "finance_profile",
        (
            "Education\nBachelor finance graduated 2024\nExperience\n"
            "Finance internship Jan 2025 - Mar 2025"
        ),
        set(),
        "insufficient_profile_evidence",
    ),
    (
        "insurance_profile",
        "Expérience\n01/2025 - 03/2025 Stage assurance, suivi des dossiers de souscription.",
        {"Insurance"},
        "insufficient_profile_evidence",
    ),
    (
        "customer_service",
        "Experience\nJun 2024 - Aug 2024 Accueil client, conseil client et relation client.",
        {"Customer Service", "Customer Relationship Management"},
        "insufficient_profile_evidence",
    ),
    (
        "operations_profile",
        "Experience\nJan 2024 - Apr 2024 Administrative operations and document processing.",
        {"Administrative Operations"},
        "insufficient_profile_evidence",
    ),
    (
        "project_coordination",
        "Profile\nProject coordination, leadership, teamwork and communication for student events.",
        {"Project Coordination", "Leadership", "Teamwork"},
        "insufficient_profile_evidence",
    ),
    (
        "cloud_context_r",
        "Skills\nR, Python, cloud data science and statistical modeling.",
        {"Data Science"},
        "insufficient_profile_evidence",
    ),
    (
        "plain_letter_r",
        "Profile\nR",
        set(),
        "insufficient_profile_evidence",
    ),
    (
        "plain_letter_c",
        "Profile\nC",
        set(),
        "insufficient_profile_evidence",
    ),
    (
        "retail_stock",
        "Experience\n06/2024 - 08/2024 Vente de produits, gestion des stocks et encaissement.",
        {"Stock Management", "Sales"},
        None,
    ),
    (
        "innovation_media",
        "Activities\nInnovation, sponsoring and médias for an association campaign.",
        {"Innovation Management", "Sponsorship", "Media Production"},
        "insufficient_profile_evidence",
    ),
    (
        "empty_sparse",
        "Profile\nMotivated candidate.",
        set(),
        "insufficient_profile_evidence",
    ),
)


def test_deterministic_cv_quality_fixture_suite() -> None:
    assert len(QUALITY_FIXTURES) == 15
    for name, text, expected_domains, abstention_reason in QUALITY_FIXTURES:
        profile = extract_cv_profile(_document(text))
        domains = set(profile.candidate.preferred_domains)
        assert expected_domains <= domains, name
        assert profile.extraction_quality.abstention_reason == abstention_reason, name
        assert "R" not in {
            skill.canonical_skill for skill in profile.skills if name == "plain_letter_r"
        }
        assert "C" not in {
            skill.canonical_skill for skill in profile.skills if name == "plain_letter_c"
        }

DOMAIN_GUARD_FIXTURES: tuple[tuple[str, str, bool], ...] = (
    (
        "real_commercial_support_profile",
        (
            "Experience\nCommercial support analyst. Soutien commercial, "
            "relation client, vente et suivi des ventes."
        ),
        True,
    ),
    (
        "sales_profile",
        (
            "Experience\nSales assistant. Vente de produits, suivi des ventes, "
            "encaissement et relation client."
        ),
        False,
    ),
    (
        "customer_support_profile",
        (
            "Experience\nCustomer service associate. Accueil client, "
            "service client and relation client."
        ),
        False,
    ),
    (
        "software_engineer_helping_customers",
        (
            "Experience\nSoftware engineer building APIs and helping customers "
            "integrate Python services."
        ),
        False,
    ),
    (
        "data_scientist_supporting_business",
        (
            "Experience\nData scientist supporting business teams "
            "with Python forecasting dashboards."
        ),
        False,
    ),
    (
        "project_manager_commercial_department",
        (
            "Experience\nProject manager collaborating with the commercial "
            "department on rollout planning."
        ),
        False,
    ),
    (
        "administrative_assistant_commercial_contacts",
        (
            "Experience\nAdministrative assistant maintaining commercial contacts "
            "and document processing."
        ),
        False,
    ),
)


def test_commercial_support_domain_requires_independent_business_evidence() -> None:
    for name, text, expected_commercial_support in DOMAIN_GUARD_FIXTURES:
        profile = extract_cv_profile(_document(text))
        domains = set(profile.candidate.preferred_domains)
        assert ("Commercial Support" in domains) is expected_commercial_support, name


def test_legitimate_sales_and_customer_support_domains_remain_available() -> None:
    sales = extract_cv_profile(
        _document(
            "Experience\nSales assistant. Vente de produits, "
            "suivi des ventes and encaissement."
        )
    )
    customer = extract_cv_profile(
        _document(
            "Experience\nCustomer service. Accueil client, "
            "service client and relation client."
        )
    )

    assert "Sales" in set(sales.candidate.preferred_domains)
    assert "Customer Service" in set(customer.candidate.preferred_domains)

def test_service_candidate_merge_preserves_richer_business_profile() -> None:
    from jobmarket.cv.attributes import extract_candidate_attributes
    from jobmarket.cv.service import _merge_candidate_attributes
    from jobmarket.skills.ontology import load_ontology

    profile = extract_cv_profile(_document(MANAGEMENT_CV))
    legacy_candidate = extract_candidate_attributes(MANAGEMENT_CV, set(), load_ontology())

    merged = _merge_candidate_attributes(profile.candidate, legacy_candidate)

    assert merged.education_status == "graduated"
    assert merged.career_level == "junior"
    assert merged.experience.internship_count >= 1
    assert "Management" in merged.preferred_domains
    assert "Customer Service" in merged.preferred_domains
