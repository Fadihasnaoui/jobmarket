"""End-to-end CV recommendation workflow tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.llm_extraction import CvExtractionResult
from jobmarket.cv.profile import JobRecommendation, ScoreComponents
from jobmarket.cv.workflow import (
    ConfirmedProfileInput,
    CvRecommendationRequest,
    format_workflow_response,
    run_cv_recommendation_workflow,
    workflow_response_to_json,
)
from jobmarket.skills.ontology import load_ontology


class _FakeSession:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


@pytest.fixture(autouse=True)
def fake_workflow_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    ontology = load_ontology()
    skill_ids = {entry.canonical: idx for idx, entry in enumerate(ontology.entries, start=1)}

    async def fake_load_skill_ids(session: object, ontology_arg: object) -> dict[str, int]:
        return skill_ids

    def fake_extract_with_fallback(
        document: object,
        *,
        ontology: object = None,
        skill_ids: object = None,
        client: object = None,
    ) -> CvExtractionResult:
        # These tests exercise the deterministic parser end-to-end; the LLM path (and
        # its network call) has its own dedicated, fully-mocked tests in
        # test_cv_llm_extraction.py. fallback_reason=None here means "bypassed for the
        # test double", not a real degradation — see
        # test_extraction_degradation_is_surfaced_not_silent for that path.
        profile = extract_cv_profile(document, ontology=ontology, skill_ids=skill_ids)
        return CvExtractionResult(profile, [], "llm")

    monkeypatch.setattr("jobmarket.cv.workflow.async_session_factory", lambda: _FakeSession())
    monkeypatch.setattr("jobmarket.cv.workflow.load_skill_ids", fake_load_skill_ids)
    monkeypatch.setattr(
        "jobmarket.cv.workflow.extract_cv_profile_with_fallback", fake_extract_with_fallback
    )


async def test_valid_txt_cv_to_recommendations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python, SQL and Data Analysis"), encoding="utf-8")
    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", _fake_recommend)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.canonical_extraction_status == "success"
    assert [item.job_id for item in response.recommendations] == [10]
    assert "Python" in response.recommendations[0].matched_skills


async def test_valid_docx_cv_to_recommendations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.docx"
    _write_docx(path, [_strong_cv_text("Python and SQL")])
    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", _fake_recommend)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.document_status == "parsed"
    assert response.recommendations


async def test_valid_pdf_cv_to_recommendations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(_minimal_pdf([_strong_cv_text("Python and SQL")]))
    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", _fake_recommend)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.profile is not None
    assert response.profile.mime_type == "application/pdf"
    assert response.recommendations[0].explanation.startswith("Recommended because")


async def test_scanned_pdf_returns_ocr_required_without_recommendations(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(_minimal_pdf(["   "]))

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.document_status == "ocr_required"
    assert response.ocr_required is True
    assert response.recommendations == []


async def test_empty_txt_returns_parse_error_without_recommendations(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.document_status == "parse_error"
    assert response.canonical_extraction_status == "parse_error"
    assert response.recommendations == []


async def test_weak_profile_gets_cautious_warning_not_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Low confidence must warn and cautiously score, never return zero results."""
    path = tmp_path / "weak.txt"
    path.write_text("Profile\nMotivated candidate.", encoding="utf-8")
    calls = 0

    async def recommend_despite_low_confidence(
        *args: object, **kwargs: object
    ) -> list[JobRecommendation]:
        nonlocal calls
        calls += 1
        return [_recommendation(job_id=10, score=80)]

    monkeypatch.setattr(
        "jobmarket.cv.workflow.recommend_jobs_for_cv", recommend_despite_low_confidence
    )

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.canonical_extraction_status == "insufficient_profile_evidence"
    assert calls == 1
    assert "insufficient_profile_evidence" in response.warnings
    assert response.recommendations
    assert response.recommendations[0].final_score == 65.0
    assert response.message is not None
    assert "cautious" in response.message.casefold()


async def test_extraction_degradation_is_surfaced_not_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A deterministic fallback (LLM failure) must be visible, never look like success."""
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python, SQL and Data Analysis"), encoding="utf-8")

    def degraded_extract(
        document: object,
        *,
        ontology: object = None,
        skill_ids: object = None,
        client: object = None,
    ) -> CvExtractionResult:
        profile = extract_cv_profile(document, ontology=ontology, skill_ids=skill_ids)
        return CvExtractionResult(
            profile,
            [],
            "deterministic_fallback",
            "bad_request (likely json_validate_failed): json_validate_failed",
        )

    monkeypatch.setattr("jobmarket.cv.workflow.extract_cv_profile_with_fallback", degraded_extract)
    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", _fake_recommend)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.profile is not None
    assert response.profile.extraction_method == "deterministic_fallback"
    assert response.profile.extraction_degraded is True
    assert "json_validate_failed" in (response.profile.extraction_degraded_reason or "")
    assert "llm_extraction_degraded" in response.warnings
    assert response.message is not None
    assert "LLM-based extraction failed" in response.message
    assert response.recommendations  # a degraded extraction must still get matched
    rendered = format_workflow_response(response)
    assert "DEGRADED" in rendered


async def test_out_of_scope_non_tech_profile_skips_matching_with_clear_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A genuine non-tech CV must get an honest message, not 10 low-signal tech rows."""
    path = tmp_path / "sales.txt"
    path.write_text(
        "Profile\nSales assistant with strong customer relationships.\n\n"
        "Experience\nJan 2023 - Dec 2023 Sales Assistant - ExampleCo\n"
        "Vente de produits, relation client, service client, encaissement.\n",
        encoding="utf-8",
    )
    calls = 0

    async def should_not_be_called(*args: object, **kwargs: object) -> list[JobRecommendation]:
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", should_not_be_called)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert calls == 0
    assert response.recommendations == []
    assert response.profile is not None
    # The only grounded skill here is a `soft_skill`-category ontology entry (CRM),
    # which by design must not count as tech evidence — see out_of_scope_reason().
    assert response.profile.canonical_skills == ["Client Relationship Management"]
    assert response.message is not None
    assert "outside the IT/tech scope" in response.message
    assert "out_of_scope_non_tech_profile" in response.warnings


async def test_thin_tech_profile_is_not_treated_as_out_of_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A junior dev CV with only a couple of skills must never trip the scope guard."""
    path = tmp_path / "junior_dev.txt"
    path.write_text(_strong_cv_text("Python and SQL"), encoding="utf-8")
    calls = 0

    async def recommend(*args: object, **kwargs: object) -> list[JobRecommendation]:
        nonlocal calls
        calls += 1
        return [_recommendation(job_id=10, score=82)]

    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", recommend)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert calls == 1
    assert response.recommendations
    assert response.message is None or "outside the IT/tech scope" not in response.message


async def test_user_confirmed_profile_allows_matching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "weak.txt"
    path.write_text("Profile\nMotivated candidate.", encoding="utf-8")
    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", _fake_recommend)

    response = await run_cv_recommendation_workflow(
        CvRecommendationRequest(
            cv_path=str(path),
            confirmed_profile=ConfirmedProfileInput(
                canonical_skills=["Python", "SQL"],
                domains=["Data Science"],
                career_level="junior",
            ),
        )
    )

    assert response.canonical_extraction_status == "success"
    assert response.recommendations


async def test_no_matching_jobs_returns_valid_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python"), encoding="utf-8")

    async def no_jobs(*args: object, **kwargs: object) -> list[JobRecommendation]:
        return []

    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", no_jobs)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert response.canonical_extraction_status == "success"
    assert response.recommendations == []


async def test_duplicate_jobs_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python"), encoding="utf-8")

    async def duplicate_jobs(*args: object, **kwargs: object) -> list[JobRecommendation]:
        item = _recommendation(job_id=10, score=80)
        return [item, item.model_copy(update={"final_score": 70})]

    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", duplicate_jobs)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert [item.job_id for item in response.recommendations] == [10]
    assert response.recommendations[0].final_score == 80


async def test_same_input_twice_has_stable_ranking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python"), encoding="utf-8")

    async def unsorted_jobs(*args: object, **kwargs: object) -> list[JobRecommendation]:
        return [_recommendation(job_id=20, score=70), _recommendation(job_id=10, score=80)]

    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", unsorted_jobs)

    first = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))
    second = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))

    assert [item.job_id for item in first.recommendations] == [10, 20]
    assert [item.job_id for item in second.recommendations] == [10, 20]


async def test_missing_embedding_fallback_path_is_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python"), encoding="utf-8")

    async def lexical_fallback(*args: object, **kwargs: object) -> list[JobRecommendation]:
        return [_recommendation(job_id=10, score=75, retrieval_source="lexical")]

    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", lexical_fallback)

    response = await run_cv_recommendation_workflow(
        CvRecommendationRequest(cv_path=str(path), mode="hybrid")
    )

    assert response.recommendations[0].job_id == 10
    assert response.diagnostics["fallback_path_used"] == "none"


def test_json_and_human_output_are_privacy_safe() -> None:
    response = _minimal_response()

    assert "SECRET" not in workflow_response_to_json(response)
    assert "SECRET" not in format_workflow_response(response)
    assert "Recommended because" in format_workflow_response(response)


async def test_explanation_uses_only_score_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv.txt"
    path.write_text(_strong_cv_text("Python"), encoding="utf-8")
    monkeypatch.setattr("jobmarket.cv.workflow.recommend_jobs_for_cv", _fake_recommend)

    response = await run_cv_recommendation_workflow(CvRecommendationRequest(cv_path=str(path)))
    explanation = response.recommendations[0].explanation

    assert "Python" in explanation
    assert "salary" not in explanation.casefold()
    assert "SECRET" not in explanation


async def _fake_recommend(*args: object, **kwargs: object) -> list[JobRecommendation]:
    return [_recommendation(job_id=10, score=82)]


def _recommendation(
    *,
    job_id: int,
    score: float,
    retrieval_source: str = "lexical",
) -> JobRecommendation:
    components = ScoreComponents(
        skill_score=0.9,
        coverage_component=1.0,
        relevance_component=0.5,
        richness_component=0.9,
        specificity_component=0.8,
        skill_profile_confidence=0.9,
        experience_score=0.8,
        career_level_score=0.9,
        job_type_score=0.5,
        domain_score=0.75,
        location_score=0.5,
        weighted_score_before_penalty=0.82,
        penalty_total=0.0,
        lexical_score=0.82,
    )
    return JobRecommendation(
        job_id=job_id,
        title="Python Data Analyst",
        company="ExampleCo",
        location="Paris, France",
        contract_type="cdi",
        source_url="https://example.test/job",
        final_score=score,
        coverage_score=1.0,
        cv_overlap_score=0.5,
        cv_skill_count=2,
        job_skill_count=3,
        matched_skill_count=2,
        matched_skills=["Python", "SQL"],
        missing_skills=["Power BI"],
        enrichment_run_id=342,
        scoring_version="cv-job-scorer-v3",
        explanation="old explanation",
        score_components=components,
        retrieval_source=retrieval_source,
    )


def _minimal_response():
    from jobmarket.cv.workflow import (
        CvRecommendationResponse,
        ExtractedProfileOutput,
        RecommendationOutput,
    )

    return CvRecommendationResponse(
        document_status="parsed",
        canonical_extraction_status="success",
        ocr_required=False,
        profile=ExtractedProfileOutput(
            filename="cv.txt",
            mime_type="text/plain",
            document_status="parsed",
            canonical_extraction_status="success",
            ocr_required=False,
            canonical_skills=["Python"],
        ),
        recommendations=[
            RecommendationOutput(
                job_id=1,
                title="Python Data Analyst",
                company="ExampleCo",
                location="Paris, France",
                contract_type="cdi",
                final_score=80,
                semantic_score=None,
                skill_score=0.9,
                role_domain_score=0.75,
                career_level_compatibility=0.9,
                matched_skills=["Python"],
                missing_important_skills=["Power BI"],
                explanation="Recommended because the CV and job share Python.",
                source_url="https://example.test/job",
            )
        ],
    )


def _strong_cv_text(skills: str) -> str:
    return (
        "Profile\nManagement-oriented data analyst with communication and public speaking.\n"
        f"Skills\n{skills}, AWS\n"
        "Education\nBachelor in management - Example University\n"
        "2018 - 2021\n"
        "Status: Graduated\n"
        "Experience\nJan 2021 - Dec 2023 Sales Assistant - ExampleCo\n"
        "Vente de produits, relation client, communication and documented delivery."
    )


_DOCX_CONTENT_TYPES = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
    "<Default Extension='rels' "
    "ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
    "<Default Extension='xml' ContentType='application/xml'/>"
    "<Override PartName='/word/document.xml' ContentType="
    "'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
    "</Types>"
)
_DOCX_ROOT_RELS = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
    "<Relationship Id='rId1' Type="
    "'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' "
    "Target='word/document.xml'/>"
    "</Relationships>"
)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    """A minimal but spec-valid OPC package — python-docx also requires
    `[Content_Types].xml` and `_rels/.rels`, not just `word/document.xml`."""
    body = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    xml = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_ROOT_RELS)
        archive.writestr("word/document.xml", xml)


def _minimal_pdf(page_texts: list[str]) -> bytes:
    objects = ["1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj"]
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(page_texts)))
    objects.append(f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >> endobj")
    for index, text in enumerate(page_texts):
        page_id = 3 + index * 2
        content_id = page_id + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects.append(
            f"{page_id} 0 obj << /Type /Page /Parent 2 0 R /Resources "
            f"<< /Font << /F1 99 0 R >> >> /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R >> endobj"
        )
        objects.append(
            f"{content_id} 0 obj << /Length {len(stream.encode())} >> stream\n"
            f"{stream}\nendstream endobj"
        )
    objects.append("99 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    content = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(content.encode()))
        content += obj + "\n"
    xref_offset = len(content.encode())
    max_obj = 99
    entries = ["0000000000 65535 f "] + ["0000000000 00000 f "] * max_obj
    for obj, offset in zip(objects, offsets[1:], strict=True):
        obj_id = int(obj.split()[0])
        entries[obj_id] = f"{offset:010d} 00000 n "
    content += f"xref\n0 {max_obj + 1}\n" + "\n".join(entries) + "\n"
    content += f"trailer << /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    return content.encode()
