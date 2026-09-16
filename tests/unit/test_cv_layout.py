"""Regression tests for layout-aware visual CV extraction."""

from __future__ import annotations

from jobmarket.cv.documents import _reconstruct_page_reading_order, order_layout_blocks
from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.profile import DocumentPage, DocumentTextBlock, ParsedDocument


def test_layout_order_keeps_sidebar_and_main_columns_deterministic() -> None:
    blocks = _visual_cv_blocks()

    ordered = order_layout_blocks(blocks)

    assert [block.block_id for block in ordered[:4]] == [
        "p1-sidebar-profile",
        "p1-sidebar-profile-body",
        "p1-sidebar-language",
        "p1-sidebar-language-body",
    ]
    assert ordered[-1].block_id == "p1-main-skills"


def test_layout_order_handles_y_increasing_downward() -> None:
    """Some PDF generators (confirmed: Chromium/Playwright print-to-PDF, the
    engine behind many browser-based resume-builder templates) emit block
    y-coordinates that increase toward the BOTTOM of the page, the opposite of
    the traditional bottom-left-origin PDF convention `_visual_cv_blocks()`
    above assumes. Getting this wrong previously reversed/scrambled the entire
    reading order for that whole class of real CVs -- see docs/LIMITATIONS.md.
    """
    blocks = _downward_y_cv_blocks()

    ordered = order_layout_blocks(blocks)

    assert [block.block_id for block in ordered] == [
        "p1-name",
        "p1-title",
        "p1-section-header",
        "p1-job-company",
        "p1-job-bullet-1",
        "p1-job-bullet-2",
        "p1-sidebar-header",
        "p1-sidebar-item-1",
        "p1-sidebar-item-2",
    ]


def test_reading_order_keeps_same_row_content_on_one_line() -> None:
    """A title on the left and a right-flush date range (or an inline
    "Title, Company, Location" one-liner followed by a right-flush date on the
    same visual row) must land on ONE reconstructed line, not be torn apart into
    separate lines just because they're far apart in x -- the exact real-CV
    pattern (compact "PREVIOUS EXPERIENCE" one-liners) that previously caused
    titles to cross-pair with the wrong company. See docs/LIMITATIONS.md.
    """
    # Fewer than 6 blocks, so `_y_increases_downward` defaults to the standard
    # PDF convention (higher y = further up the page) -- rows are given
    # decreasing y accordingly. Includes a section header with no date
    # counterpart, so the "title" column isn't row-for-row identical to the
    # "date" column -- reattachment must tell a genuine content column (however
    # short) apart from a pure metadata column by that asymmetry, not by size.
    blocks = [
        _block("p1-section-header", 1, 24, 720, "PREVIOUS EXPERIENCE"),
        _block("p1-title", 1, 24, 700, "Junior Programmer, ABC Company, London, UK"),
        _block("p1-date", 1, 480, 700, "06/2017 - 10/2018"),
        _block("p1-title-2", 1, 24, 680, "SQL DBA, XYZ Company, New York, USA"),
        _block("p1-date-2", 1, 480, 680, "01/2016 - 05/2017"),
    ]

    text = _reconstruct_page_reading_order(blocks)

    lines = text.splitlines()
    assert lines == [
        "PREVIOUS EXPERIENCE",
        "Junior Programmer, ABC Company, London, UK 06/2017 - 10/2018",
        "SQL DBA, XYZ Company, New York, USA 01/2016 - 05/2017",
    ]


def test_visual_management_pdf_layout_extracts_business_profile() -> None:
    document = _layout_document(_visual_cv_blocks())

    profile = extract_cv_profile(document)

    labels = {section.normalized_label for section in profile.sections}
    assert profile.extraction_quality.section_coverage > 0
    assert {"profile", "education", "experience", "activities", "languages"} <= labels
    assert profile.education_detail is not None
    assert profile.education_detail.education_level == "bachelor/licence"
    assert profile.education_detail.education_field == "management"
    assert profile.education_detail.education_status == "graduated"
    assert profile.education_detail.institution == "ESSECT"
    assert len(profile.experience_entries) >= 3
    assert profile.candidate.career_level == "junior"
    domains = set(profile.candidate.preferred_domains)
    assert {"Management", "Sales", "Customer Service", "Insurance"} <= domains
    assert "Communication" in domains
    assert "Digital Transformation" in domains
    assert "R" not in {skill.canonical_skill for skill in profile.skills}
    assert "C" not in {skill.canonical_skill for skill in profile.skills}
    assert profile.extraction_quality.abstention_reason is None
    assert profile.extraction_quality.extraction_confidence_label in {"medium", "high"}
    assert profile.extraction_traces


def _layout_document(blocks: list[DocumentTextBlock]) -> ParsedDocument:
    ordered = order_layout_blocks(blocks)
    text = "\n".join(block.text for block in ordered)
    adjusted: list[DocumentTextBlock] = []
    cursor = 0
    for block in ordered:
        start = text.index(block.text, cursor)
        end = start + len(block.text)
        adjusted.append(block.model_copy(update={"start_offset": start, "end_offset": end}))
        cursor = end
    return ParsedDocument(
        filename="visual-cv.pdf",
        mime_type="application/pdf",
        text=text,
        page_count=1,
        pages=[DocumentPage(page=1, text=text, start_offset=0, end_offset=len(text))],
        blocks=adjusted,
        warnings=[],
        file_hash="a" * 64,
        file_size=len(text.encode()),
    )


def _visual_cv_blocks() -> list[DocumentTextBlock]:
    return [
        _block("p1-sidebar-profile", 1, 42, 740, "Profil"),
        _block(
            "p1-sidebar-profile-body",
            1,
            42,
            710,
            "Jeune dipl?m?e Licence en Management, communication, leadership, "
            "transformation digitale.",
        ),
        _block("p1-sidebar-language", 1, 42, 650, "Language"),
        _block("p1-sidebar-language-body", 1, 42, 625, "Fran?ais Anglais"),
        _block("p1-sidebar-life", 1, 42, 570, "Vie Associative"),
        _block("p1-sidebar-life-body", 1, 42, 545, "Sponsoring et organisation ?v?nementielle"),
        _block("p1-main-education", 1, 250, 740, "Education"),
        _block("p1-main-degree", 1, 250, 712, "Licence en management - ESSECT - dipl?m?e 2025"),
        _block("p1-main-experience", 1, 250, 660, "Exp?rience professionnelle"),
        _block("p1-date-sales", 1, 175, 625, "06-2024 / 08-2024"),
        _block("p1-main-sales", 1, 250, 625, "Vendeuse - Parfumerie"),
        _block(
            "p1-main-sales-body",
            1,
            250,
            602,
            "Customer service, accueil client, vente de produits, gestion des stocks.",
        ),
        _block("p1-date-insurance", 1, 175, 550, "01-06-2025 / 31-07-2025"),
        _block("p1-main-insurance", 1, 250, 550, "Stagiaire - GAT Assurances"),
        _block(
            "p1-main-insurance-body",
            1,
            250,
            527,
            "Dossiers de souscription assurance et contr?le des dossiers administratifs.",
        ),
        _block("p1-date-ef", 1, 175, 475, "Juin 2026 / Pr?sent"),
        _block("p1-main-ef", 1, 250, 475, "Education First"),
        _block(
            "p1-main-ef-body",
            1,
            250,
            452,
            "Commercial support, relation client, accompagnement des ?tudiants.",
        ),
        _block("p1-main-formation", 1, 250, 390, "Formation"),
        _block("p1-main-skills", 1, 250, 365, "Communication Leadership Digital transformation"),
    ]


def _downward_y_cv_blocks() -> list[DocumentTextBlock]:
    # Coordinates and content-stream order shaped after a real Chromium-rendered
    # CV PDF: the main column is fully emitted top-to-bottom (y increasing) before
    # the sidebar, which itself restarts from a low y (a separate positioned box
    # near the top of the page, not a continuation of the main column's y range).
    return [
        _block("p1-name", 1, 24, 40, "First Last"),
        _block("p1-title", 1, 24, 59, "Data Scientist"),
        _block("p1-section-header", 1, 24, 86, "WORK EXPERIENCE"),
        _block("p1-job-company", 1, 24, 109, "Resume Worded, London, United Kingdom"),
        _block("p1-job-bullet-1", 1, 24, 130, "Increased the usage and adoption of AI."),
        _block("p1-job-bullet-2", 1, 24, 145, "Designed an anomaly detection framework."),
        _block("p1-sidebar-header", 1, 650, 50, "CONTACT"),
        _block("p1-sidebar-item-1", 1, 650, 70, "Preston, United Kingdom"),
        _block("p1-sidebar-item-2", 1, 650, 90, "first.last@gmail.com"),
    ]


def _block(block_id: str, page: int, x: float, y: float, text: str) -> DocumentTextBlock:
    return DocumentTextBlock(
        page=page,
        block_id=block_id,
        x=x,
        y=y,
        width=max(20.0, len(text) * 5.0),
        height=12.0,
        text=text,
    )
