"""Tests for LLM-based CV extraction. All LLM calls mocked — zero real network access."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from openai import BadRequestError

from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.llm_extraction import (
    CvLlmExtractionError,
    call_llm_cv_extraction,
    extract_cv_profile_llm,
    extract_cv_profile_with_fallback,
)
from jobmarket.cv.llm_guard import validate_llm_skill_candidates
from jobmarket.cv.profile import DocumentPage, ParsedDocument
from jobmarket.skills.ontology import load_ontology

CV_TEXT = (
    "Profile\nSoftware engineering student.\n\n"
    "Experience\nJan 2024 - Jun 2024 Backend Engineering Intern at Example Corp. "
    "Built APIs with Python and deployed on AWS.\n\n"
    "Education\nMaster in Computer Science, Example University, graduated 2025.\n\n"
    "Skills\nPython, AWS, Kubernetes.\n"
)


def _mock_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


def _valid_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "skills": [
            {"name": "Python", "evidence_quote": "Python", "confidence": 0.9},
            {"name": "AWS", "evidence_quote": "AWS", "confidence": 0.85},
        ],
        "experiences": [
            {
                "title": "Backend Engineering Intern",
                "company": "Example Corp",
                "start_date": "Jan 2024",
                "end_date": "Jun 2024",
                "duration_months": 6,
                "is_internship": True,
                "domain": None,
                "evidence_quote": (
                    "Jan 2024 - Jun 2024 Backend Engineering Intern at Example Corp"
                ),
                "impact_evidence_quote": "Built APIs with Python and deployed on AWS.",
            }
        ],
        "education": [
            {
                "level": "master/mba",
                "field": "computer science",
                "institution": "Example University",
                "graduation_year": 2025,
                "status": "graduated",
                "evidence_quote": "Master in Computer Science, Example University",
            }
        ],
        "role_families": ["Software Engineering"],
        "domains": ["Software Engineering"],
        "seniority": "internship",
        "total_years_experience": 0.5,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _document(text: str = CV_TEXT) -> ParsedDocument:
    return ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text=text,
        page_count=None,
        pages=[DocumentPage(page=None, text=text, start_offset=0, end_offset=len(text))],
        warnings=[],
        file_hash="a" * 64,
        file_size=len(text.encode()),
    )


def test_call_llm_cv_extraction_parses_valid_json() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())

    result = call_llm_cv_extraction(CV_TEXT, client=client)

    assert [skill.name for skill in result.skills] == ["Python", "AWS"]
    assert result.seniority == "internship"
    assert result.experiences[0].company == "Example Corp"
    assert client.chat.completions.create.call_count == 1


def test_call_llm_cv_extraction_retries_once_on_malformed_json() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("not json"),
        _mock_response(_valid_payload()),
    ]

    result = call_llm_cv_extraction(CV_TEXT, client=client)

    assert result.seniority == "internship"
    assert client.chat.completions.create.call_count == 2


def test_call_llm_cv_extraction_requests_raised_max_completion_tokens() -> None:
    """Regression: a real CV hit Groq's json_validate_failed with failed_generation

    "max completion tokens reached before generating a valid document" — no cap was
    ever being passed before, so the provider default applied and was too low. Both
    the first attempt and the simplified retry share this one call site, so a single
    kwarg change covers both.
    """
    from jobmarket.cv.llm_extraction import MAX_COMPLETION_TOKENS

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("not json"),
        _mock_response(_valid_payload()),
    ]

    call_llm_cv_extraction(CV_TEXT, client=client)

    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["max_completion_tokens"] == MAX_COMPLETION_TOKENS


def test_call_llm_cv_extraction_raises_after_max_attempts() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("still not json")

    try:
        call_llm_cv_extraction(CV_TEXT, client=client)
        raise AssertionError("expected CvLlmExtractionError")
    except CvLlmExtractionError:
        pass
    assert client.chat.completions.create.call_count == 2


def test_call_llm_cv_extraction_wraps_client_errors() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("boom")

    try:
        call_llm_cv_extraction(CV_TEXT, client=client)
        raise AssertionError("expected CvLlmExtractionError")
    except CvLlmExtractionError as exc:
        assert "boom" in str(exc)


def test_extract_cv_profile_llm_grounds_ontology_skills_and_facts() -> None:
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    document = _document()

    draft = extract_cv_profile_llm(document, ontology=ontology, skill_ids=skill_ids, client=client)

    canonicals = {c["canonical_skill"] for c in draft.skill_candidates}
    assert canonicals == {"Python", "AWS"}
    for candidate in draft.skill_candidates:
        start, end = candidate["document_start"], candidate["document_end"]
        assert document.text[start:end] == candidate["evidence_quote"]
    assert draft.profile.experience_entries[0].entry_type == "internship"
    assert draft.profile.experience_entries[0].employer == "Example Corp"
    assert draft.profile.experience_entries[0].impact_evidence == (
        "Built APIs with Python and deployed on AWS."
    )
    assert draft.profile.education_detail is not None
    assert draft.profile.education_detail.education_level == "master/mba"
    assert draft.profile.candidate.career_level == "internship"
    assert {"Python", "AWS", "Kubernetes"}.issubset(
        {skill.canonical_skill for skill in draft.profile.skills}
    )


def test_llm_experience_dates_are_recovered_from_grounded_cv_context() -> None:
    """Date ranges next to a grounded role must win over incomplete LLM fields."""
    cv_text = (
        "Resume Worded, New York, NY\nData Scientist\nJanuary 2018 - Present\n"
        "Built data products.\n\n"
        "Growthsi, Boston, MA\nAssociate Data Scientist\nJuly 2015 - January 2018\n"
        "Built forecasting models.\n\n"
        "ACB Corporation Inc., Boston, MA\nData Scientist Intern\nJanuary 2015 - June 2015\n"
        "Supported analysis."
    )
    document = _document(cv_text)
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            skills=[],
            education=[],
            experiences=[
                {
                    "title": "Data Scientist",
                    "company": "Resume Worded",
                    "start_date": None,
                    "end_date": "Present",
                    "duration_months": None,
                    "is_internship": False,
                    "domain": "Data Science",
                    "evidence_quote": "Resume Worded, New York, NY\nData Scientist",
                },
                {
                    "title": "Associate Data Scientist",
                    "company": "Growthsi",
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": "Data Science",
                    "evidence_quote": "Growthsi, Boston, MA\nAssociate Data Scientist",
                },
                {
                    "title": "Data Scientist Intern",
                    "company": "ACB Corporation Inc.",
                    "start_date": "January 2015",
                    "end_date": "June 2015",
                    "duration_months": 3,
                    "is_internship": True,
                    "domain": "Data Science",
                    "evidence_quote": "ACB Corporation Inc., Boston, MA\nData Scientist Intern",
                },
            ],
        )
    )
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}

    draft = extract_cv_profile_llm(document, ontology=ontology, skill_ids=skill_ids, client=client)
    durations = {entry.title: entry.duration_months for entry in draft.profile.experience_entries}

    expected_current_months = (date.today().year - 2018) * 12 + date.today().month
    assert durations == {
        "Data Scientist": expected_current_months,
        "Associate Data Scientist": 31,
        "Data Scientist Intern": 6,
    }


def test_networking_skills_survive_the_llm_ontology_guard() -> None:
    """Regression: a network-engineering CV must not lose its grounded skills.

    The mocked LLM returns skills that occur verbatim in the CV.  The assertion covers
    the exact LLM -> ontology -> evidence-guard path used by the upload endpoint.
    """
    raw_skills = [
        "TCP/IP", "BGP", "OSPF", "EIGRP", "SD-WAN", "DNS", "DHCP",
        "Firewalls", "Palo Alto", "Cisco ASA", "VPN", "IDS/IPS", "ACLs",
        "AWS VPC", "Azure VNet", "Wireshark", "SolarWinds", "Nagios",
        "LAN/WAN", "Load Balancing",
    ]
    document = _document(
        "Mason Thomas\nNetwork Engineer\nSkills: " + ", ".join(raw_skills)
    )
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    # Simulate the observed failure: the LLM returns only two items from the
    # visible skills list. The deterministic exact-text matches must retain the rest.
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            skills=[
                {"name": skill, "evidence_quote": skill, "confidence": 0.95}
                for skill in ["LAN/WAN", "Load Balancing"]
            ],
            experiences=[],
            education=[],
        )
    )

    draft = extract_cv_profile_llm(document, ontology=ontology, skill_ids=skill_ids, client=client)
    report = validate_llm_skill_candidates(
        document,
        draft.skill_candidates,
        deterministic_skills=draft.profile.skills,
        ontology=ontology,
        skill_ids=skill_ids,
    )

    # The two LLM candidates duplicate deterministic exact-text matches; the guard
    # rejects only those duplicates while retaining every visible skill.
    assert {item.reason for item in report.rejected_candidates} == {
        "Canonical skill already exists"
    }
    assert {
        "TCP/IP", "BGP", "OSPF", "EIGRP", "SD-WAN", "DNS", "DHCP",
        "Firewalls", "Palo Alto Networks", "Cisco ASA", "VPN", "IDS/IPS",
        "Access Control Lists", "AWS VPC", "Azure Virtual Network", "Wireshark",
        "SolarWinds", "Nagios", "LAN/WAN", "Load Balancing",
    }.issubset({skill.canonical_skill for skill in report.final_skills})


def test_extract_cv_profile_llm_drops_unmapped_and_hallucinated_skills() -> None:
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            skills=[
                {"name": "Python", "evidence_quote": "Python", "confidence": 0.9},
                {"name": "TotallyMadeUpSkill", "evidence_quote": "Python", "confidence": 0.9},
                {
                    "name": "Rust",
                    "evidence_quote": "never appears in the CV text",
                    "confidence": 0.9,
                },
            ]
        )
    )
    document = _document()

    draft = extract_cv_profile_llm(document, ontology=ontology, skill_ids=skill_ids, client=client)

    canonicals = {c["canonical_skill"] for c in draft.skill_candidates}
    assert canonicals == {"Python"}


def test_fabricated_internship_does_not_inflate_grounded_internship_count() -> None:
    """Regression: a fabricated internship (no title/company grounding) must not survive.

    Before the first fix, internship_count summed ALL LLM-claimed internship entries
    regardless of grounding. Before this second fix, an ungrounded entry was kept in
    `experience_entries` (just at lower confidence) instead of being dropped outright.
    A fabricated second internship — title and company that appear nowhere in the CV,
    even though its evidence_quote is a real (but misleading) substring — must now be
    rejected entirely, not merely down-weighted.
    """
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "title": "Backend Engineering Intern",
                    "company": "Example Corp",
                    "start_date": "Jan 2024",
                    "end_date": "Jun 2024",
                    "duration_months": 6,
                    "is_internship": True,
                    "domain": None,
                    "evidence_quote": (
                        "Jan 2024 - Jun 2024 Backend Engineering Intern at Example Corp"
                    ),
                },
                {
                    "title": "Data Science Intern",
                    "company": "Fabricated Corp",
                    "start_date": "Jul 2023",
                    "end_date": "Dec 2023",
                    "duration_months": 6,
                    "is_internship": True,
                    "domain": None,
                    "evidence_quote": (
                        "Jul 2023 - Dec 2023 Data Science Intern at Fabricated Corp"
                    ),
                },
            ]
        )
    )
    document = _document()

    draft = extract_cv_profile_llm(document, ontology=ontology, skill_ids=skill_ids, client=client)

    assert len(draft.profile.experience_entries) == 1
    assert draft.profile.experience_entries[0].employer == "Example Corp"
    assert draft.profile.candidate.experience.internship_count == 1


_SCRAMBLED_RESUME_TEMPLATE_CV = (
    "First Last\nData Scientist\n"
    "Resume Worded,\nData Scientist\n"
    "Polyhire,\nStatistical Programmer\n"
    "Growthsi,\nDatabase Developer\n"
    "Junior Programmer,\nSQL DBA,\nSystem Admin (Internship),\n"
    "University of New York\n"
    "● Increased the usage and adoption of AI in 20+ departments\n"
    "● Designed an anomaly detection framework for 10+ digital channels of RW\n"
    "● Improved customer retention by creating a personalized statistical formula\n"
    "● Designed a database to track purchase orders, invoices, and shipping\n"
    "ABC Company, London, UK\n06/2017\n"
    "XYZ Company, New York, USA\n01/2016 - 05/2017\n"
    "ABC, New York, USA\n07/2014 - 10/2018\n"
    "WORK EXPERIENCE\nEDUCATION\nBachelor of Science, Information and Data Science\n"
    "OTHER\nPrincipal Data Scientist (PDS)\nCertified Analytics Professional (CAP)\n"
)


def test_experience_extraction_drops_phantom_bullets_and_certifications() -> None:
    """Regression for the 3 defects found on a real Data-Scientist/Resume-Worded CV.

    A resume-builder template got its layout flattened to scrambled plain text
    (companies, titles, bullets, and dates each landed in their own block instead of
    staying grouped per job) — a real, previously-observed extraction pattern, not a
    contrived one. Simulates the buggy LLM behavior this fix targets: bullets split
    into phantom roles (no title/company, just a bullet's own text as evidence_quote,
    which is trivially "grounded" since it's real CV text) and certifications pulled
    in as jobs. Asserts the post-processing guard leaves exactly the 6 real jobs,
    correctly titled, with no phantom or certification entries surviving.
    """
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "title": "Data Scientist",
                    "company": "Resume Worded",
                    "start_date": None,
                    "end_date": "Present",
                    "duration_months": None,
                    "is_internship": False,
                    "domain": "Data Science",
                    "evidence_quote": "Resume Worded,\nData Scientist",
                },
                {
                    "title": "Statistical Programmer",
                    "company": "Polyhire",
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": "Polyhire,\nStatistical Programmer",
                },
                {
                    "title": "Database Developer",
                    "company": "Growthsi",
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": "Growthsi,\nDatabase Developer",
                },
                {
                    "title": "Junior Programmer",
                    "company": "ABC Company",
                    "start_date": None,
                    "end_date": "06/2017",
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": "ABC Company, London, UK",
                },
                {
                    "title": "SQL DBA",
                    "company": "XYZ Company",
                    "start_date": "01/2016",
                    "end_date": "05/2017",
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": "XYZ Company, New York, USA",
                },
                {
                    "title": "System Admin (Internship)",
                    "company": "ABC",
                    "start_date": "07/2014",
                    "end_date": "10/2018",
                    "duration_months": None,
                    "is_internship": True,
                    "domain": None,
                    "evidence_quote": "ABC, New York, USA",
                },
                # Phantom entries a badly-behaved LLM might still emit from bullets:
                # no title, no company — only the bullet's own (trivially real) text.
                {
                    "title": None,
                    "company": None,
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": (
                        "Increased the usage and adoption of AI in 20+ departments"
                    ),
                },
                {
                    "title": None,
                    "company": None,
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": (
                        "Designed an anomaly detection framework for 10+ digital channels of RW"
                    ),
                },
                # Certifications a badly-behaved LLM might still pull in as jobs.
                {
                    "title": "Principal Data Scientist (PDS)",
                    "company": None,
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": "Principal Data Scientist (PDS)",
                },
                {
                    "title": "Certified Analytics Professional (CAP)",
                    "company": None,
                    "start_date": None,
                    "end_date": None,
                    "duration_months": None,
                    "is_internship": False,
                    "domain": None,
                    "evidence_quote": "Certified Analytics Professional (CAP)",
                },
            ]
        )
    )
    document = _document(_SCRAMBLED_RESUME_TEMPLATE_CV)

    draft = extract_cv_profile_llm(document, ontology=ontology, skill_ids=skill_ids, client=client)
    entries = draft.profile.experience_entries

    assert len(entries) == 6
    pairs = {(entry.title, entry.employer) for entry in entries}
    assert pairs == {
        ("Data Scientist", "Resume Worded"),
        ("Statistical Programmer", "Polyhire"),
        ("Database Developer", "Growthsi"),
        ("Junior Programmer", "ABC Company"),
        ("SQL DBA", "XYZ Company"),
        ("System Admin (Internship)", "ABC"),
    }
    assert all(entry.title is not None for entry in entries)  # no "Untitled role"
    titles = {entry.title for entry in entries}
    assert "Principal Data Scientist (PDS)" not in titles
    assert "Certified Analytics Professional (CAP)" not in titles
    assert sum(1 for entry in entries if entry.entry_type == "internship") == 1


def test_extract_cv_profile_llm_raises_on_empty_groundable_content() -> None:
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        json.dumps(
            {
                "skills": [],
                "experiences": [],
                "education": [],
                "role_families": [],
                "domains": [],
                "seniority": "unknown",
                "total_years_experience": None,
            }
        )
    )

    try:
        extract_cv_profile_llm(
            _document("Profile\nMotivated candidate."),
            ontology=ontology,
            skill_ids=skill_ids,
            client=client,
        )
        raise AssertionError("expected CvLlmExtractionError")
    except CvLlmExtractionError:
        pass


def test_extract_cv_profile_with_fallback_uses_llm_result_on_success() -> None:
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    document = _document()

    result = extract_cv_profile_with_fallback(
        document, ontology=ontology, skill_ids=skill_ids, client=client
    )

    assert result.extraction_method == "llm"
    assert result.fallback_reason is None
    assert {"Python", "AWS", "Kubernetes"}.issubset(
        {skill.canonical_skill for skill in result.profile.skills}
    )
    assert result.skill_candidates


def test_extract_cv_profile_with_fallback_falls_back_on_llm_failure() -> None:
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("network down")
    document = _document()

    result = extract_cv_profile_with_fallback(
        document, ontology=ontology, skill_ids=skill_ids, client=client
    )
    expected = extract_cv_profile(document, ontology=ontology, skill_ids=skill_ids)

    assert result.extraction_method == "deterministic_fallback"
    assert result.skill_candidates == []
    assert result.fallback_reason is not None
    assert "network down" in result.fallback_reason
    assert {skill.canonical_skill for skill in result.profile.skills} == {
        skill.canonical_skill for skill in expected.skills
    }


def test_call_llm_cv_extraction_retries_bad_request_with_simplified_prompt() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        BadRequestError(
            message="json_validate_failed",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        ),
        _mock_response(_valid_payload()),
    ]

    result = call_llm_cv_extraction(CV_TEXT, client=client)

    assert result.seniority == "internship"
    assert client.chat.completions.create.call_count == 2
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_call_messages[0]["role"] == "system"
    assert "ONLY" in second_call_messages[0]["content"]


def test_call_llm_cv_extraction_raises_after_repeated_bad_request() -> None:
    client = MagicMock()
    error = BadRequestError(
        message="json_validate_failed",
        response=MagicMock(status_code=400, headers={}),
        body=None,
    )
    client.chat.completions.create.side_effect = [error, error]

    try:
        call_llm_cv_extraction(CV_TEXT, client=client)
        raise AssertionError("expected CvLlmExtractionError")
    except CvLlmExtractionError as exc:
        assert "json_validate_failed" in str(exc)
    assert client.chat.completions.create.call_count == 2


def test_call_llm_cv_extraction_truncates_to_max_cv_chars() -> None:
    from jobmarket.cv.llm_extraction import MAX_CV_CHARS

    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    long_text = "x" * (MAX_CV_CHARS + 2000)

    call_llm_cv_extraction(long_text, client=client)

    sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "x" * (MAX_CV_CHARS + 1) not in user_content
    assert "x" * MAX_CV_CHARS in user_content
