# ruff: noqa: E501
"""CLI entry point."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

import typer

from jobmarket.cv.documents import CvDocumentError
from jobmarket.cv.matching import DEFAULT_RECOMMENDATION_RUN_ID, RecommendationMode
from jobmarket.cv.reporting import cv_match_result_to_json, format_cv_match_result
from jobmarket.cv.service import match_cv_document
from jobmarket.cv.workflow import (
    ConfirmedProfileInput,
    CvRecommendationRequest,
    RecommendationFilters,
    format_workflow_response,
    run_cv_recommendation_workflow,
    workflow_response_to_json,
)
from jobmarket.datasets.annotation import prepare_gold_structure
from jobmarket.datasets.audit import write_audit_reports
from jobmarket.datasets.duplicates import write_duplicate_report
from jobmarket.datasets.evaluator import evaluate_cv_extraction
from jobmarket.datasets.gold_review import (
    format_review_item,
    format_review_progress,
    generate_manual_review_package,
    generate_prefilled_final_review,
    review_progress,
    safe_review_item,
    validate_final_review,
)
from jobmarket.datasets.inventory import write_inventory_reports
from jobmarket.datasets.registry import generate_registry
from jobmarket.datasets.splits import validate_splits
from jobmarket.datasets.synthetic_gold import (
    evaluate_synthetic_cv_extraction,
    generate_synthetic_gold,
    validate_synthetic_gold,
)
from jobmarket.embeddings.runner import JobEmbeddingConfig, run_job_embedding
from jobmarket.ingest.registry import all_source_names, get_source
from jobmarket.ingest.runner import run_ingest
from jobmarket.llm_benchmark import DEFAULT_OUTPUT_DIR, run_groq_benchmark
from jobmarket.logging_config import configure_logging
from jobmarket.parse.runner import run_parse
from jobmarket.reporting.dedup import collect_dedup_report, format_dedup_report
from jobmarket.reporting.enrichment import (
    EnrichmentRunReport,
    collect_enrichment_report,
    enrichment_report_to_json,
    format_enrichment_report,
)
from jobmarket.reporting.stats import collect_stats, format_stats
from jobmarket.skills.enrichment import (
    EnrichmentConfigError,
    MatcherEnrichmentConfig,
    MatcherEnrichmentSummary,
    OntologySyncError,
    run_matcher_enrichment,
)
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology
from jobmarket.skills.sync import sync_skills

app = typer.Typer(help="Job market data ingestion pipeline.")
skills_app = typer.Typer(help="Skills ontology commands.")
enrichment_app = typer.Typer(help="Deterministic enrichment commands.")
cv_app = typer.Typer(help="CV parsing and job recommendation commands.")
embed_app = typer.Typer(help="Semantic embedding commands.")
llm_app = typer.Typer(help="Optional LLM benchmark commands.")
dataset_app = typer.Typer(help="Dataset management commands.")
evaluate_app = typer.Typer(help="Evaluation commands.")
app.add_typer(skills_app, name="skills")
app.add_typer(enrichment_app, name="enrichment")
app.add_typer(cv_app, name="cv")
app.add_typer(embed_app, name="embed")
app.add_typer(llm_app, name="llm")
app.add_typer(dataset_app, name="dataset")
app.add_typer(evaluate_app, name="evaluate")



@app.callback()
def main() -> None:
    """Initialize logging before any command runs."""
    configure_logging()


@app.command("ingest")
def ingest_cmd(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Source to ingest (e.g. adzuna).",
    ),
    all_sources: bool = typer.Option(
        False,
        "--all",
        help="Ingest all registered sources.",
    ),
) -> None:
    """Fetch raw job payloads and store them verbatim."""
    if all_sources:
        names = all_source_names()
    elif source is not None:
        names = [source]
    else:
        raise typer.BadParameter("Provide --source or --all.")

    for name in names:
        typer.echo(f"Ingesting source: {name}")
        ingest_source = get_source(name)
        asyncio.run(run_ingest(ingest_source))
        typer.echo(f"Finished: {name}")


@app.command("parse")
def parse_cmd(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Only parse rows from this source.",
    ),
    reparse: bool = typer.Option(
        False,
        "--reparse",
        help="Reset parsed_at for the given source and re-run parsing.",
    ),
) -> None:
    """Parse raw payloads into normalized jobs, companies, and job_sources."""
    if reparse and source is None:
        raise typer.BadParameter("--reparse requires --source")

    typer.echo(f"Parsing source: {source or 'all supported'}")
    summary = asyncio.run(run_parse(source=source, reparse=reparse))
    typer.echo(f"Done: parsed={summary['parsed_count']} failed={summary['failed_count']}")


@app.command("stats")
def stats_cmd() -> None:
    """Show row counts, salary coverage, and per-source breakdown."""
    report = asyncio.run(collect_stats())
    typer.echo(format_stats(report))


@app.command("dedup-report")
def dedup_report_cmd() -> None:
    """Show deduplication metrics across sources."""
    report = asyncio.run(collect_dedup_report())
    typer.echo(format_dedup_report(report))


@dataset_app.command("inventory")
def dataset_inventory_cmd() -> None:
    """Generate privacy-safe raw dataset inventory reports."""
    reports = write_inventory_reports()
    typer.echo("Dataset inventory")
    typer.echo("json: datasets/reports/dataset_inventory.json")
    typer.echo("markdown: datasets/reports/dataset_inventory.md")
    typer.echo(f"datasets: {len(reports)}")


@dataset_app.command("audit")
def dataset_audit_cmd() -> None:
    """Generate privacy-safe recursive raw dataset audit reports."""
    report = write_audit_reports()
    typer.echo("Dataset audit")
    typer.echo("json: datasets/reports/dataset_audit.json")
    typer.echo("markdown: datasets/reports/dataset_audit.md")
    typer.echo(f"files: {report['file_count']}")
    typer.echo(f"documents: {len(report['documents'])}")
    typer.echo(f"tabular: {len(report['tabular'])}")


@dataset_app.command("duplicates")
def dataset_duplicates_cmd() -> None:
    """Generate deterministic duplicate and near-duplicate report."""
    report = write_duplicate_report()
    typer.echo("Dataset duplicates")
    typer.echo("json: datasets/reports/duplicates.json")
    typer.echo(f"sha256_duplicate_groups: {len(report['sha256_duplicates'])}")
    typer.echo(f"text_duplicate_groups: {len(report['normalized_text_duplicates'])}")
    typer.echo(f"near_duplicate_groups: {len(report['near_duplicate_groups'])}")


@dataset_app.command("registry")
def dataset_registry_cmd() -> None:
    """Generate dataset registry with unverified fields marked UNKNOWN or TO_VERIFY."""
    report = generate_registry()
    typer.echo("Dataset registry")
    typer.echo("json: datasets/registry.json")
    typer.echo(f"datasets: {len(report['datasets'])}")


@dataset_app.command("prepare-gold")
def dataset_prepare_gold_cmd() -> None:
    """Prepare gold dataset folders, schema, split files, and empty templates."""
    prepare_gold_structure()
    typer.echo("Prepared datasets/gold structure and annotation template.")


@dataset_app.command("prepare-manual-review")
def dataset_prepare_manual_review_cmd() -> None:
    """Generate privacy-safe manual gold review package files."""
    report = generate_manual_review_package()
    typer.echo("Manual gold review package")
    for key, value in report["paths"].items():
        typer.echo(f"{key}: {value}")
    typer.echo(f"selected_count: {report['selected_count']}")
    typer.echo(f"source_groups: {report['source_groups']}")



@dataset_app.command("prefill-final-review")
def dataset_prefill_final_review_cmd() -> None:
    """Generate conservative prefilled final review proposal without approval."""
    report = generate_prefilled_final_review()
    typer.echo("Prefilled manual final review")
    for key, value in report["paths"].items():
        typer.echo(f"{key}: {value}")
    typer.echo(f"rows_prefilled: {report['rows_prefilled']}")
    typer.echo(f"anonymization_likely_count: {report['anonymization_likely_count']}")
    typer.echo(f"uncertain_pii_count: {report['uncertain_pii_count']}")
    typer.echo(f"blocked_rows: {report['blocked_rows']}")


@dataset_app.command("review-item")
def dataset_review_item_cmd(
    shortlist_id: str | None = typer.Option(None, "--shortlist-id", help="Shortlist ID."),
    gold_id: str | None = typer.Option(None, "--gold-id", help="Proposed gold ID."),
) -> None:
    """Display privacy-safe metadata for one proposed gold review item."""
    try:
        item = safe_review_item(shortlist_id=shortlist_id, gold_id=gold_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_review_item(item))



@dataset_app.command("review-progress")
def dataset_review_progress_cmd(
    review: Annotated[
        Path,
        typer.Option("--review", help="Manual review CSV to summarize."),
    ] = Path("datasets/reports/manual_gold_final_review_prefilled.csv"),
) -> None:
    """Report manual review completion progress without promotion."""
    typer.echo(format_review_progress(review_progress(review)))


@dataset_app.command("validate-final-review")
def dataset_validate_final_review_cmd(
    review: Annotated[
        Path,
        typer.Option("--review", help="Final manual review CSV to validate."),
    ] = Path("datasets/reports/manual_gold_final_review.csv"),
) -> None:
    """Validate final manual review decisions without promoting documents."""
    result = validate_final_review(review_path=review)
    typer.echo("Final review validation")
    typer.echo("json: datasets/reports/final_review_validation.json")
    typer.echo("markdown: datasets/reports/final_review_validation.md")
    typer.echo(f"valid: {result['valid']}")
    typer.echo(f"split_counts: {result['split_counts']}")
    if not result["valid"]:
        typer.echo(f"errors: {len(result['errors'])}")
        raise typer.Exit(code=1)


@dataset_app.command("generate-synthetic-gold")
def dataset_generate_synthetic_gold_cmd() -> None:
    """Generate project-owned scenario-first synthetic CV gold set."""
    report = generate_synthetic_gold()
    typer.echo("Synthetic gold generated")
    typer.echo(f"root: {report['root']}")
    typer.echo(f"documents: {report['generated_documents']}")


@dataset_app.command("validate-synthetic-gold")
def dataset_validate_synthetic_gold_cmd() -> None:
    """Validate project-owned synthetic CV gold set."""
    result = validate_synthetic_gold()
    typer.echo("Synthetic gold validation")
    typer.echo("json: datasets/gold_synthetic/reports/validation.json")
    typer.echo("markdown: datasets/gold_synthetic/reports/validation.md")
    typer.echo(f"valid: {result.valid}")
    typer.echo(f"errors: {len(result.errors)}")
    if not result.valid:
        raise typer.Exit(code=1)


@dataset_app.command("validate-splits")
def dataset_validate_splits_cmd() -> None:
    """Validate gold split files against documents and manual annotations."""
    report = validate_splits()
    typer.echo(f"valid: {report['valid']}")
    if not report["valid"]:
        typer.echo(f"duplicate_cv_between_splits: {report['duplicate_cv_between_splits']}")
        typer.echo(f"missing_annotations: {report['missing_annotations']}")
        typer.echo(f"missing_documents: {report['missing_documents']}")
        typer.echo(f"schema_errors: {report['schema_errors']}")
        raise typer.Exit(code=1)


@evaluate_app.command("cv-extraction")
def evaluate_cv_extraction_cmd(
    gold_root: Annotated[
        Path | None,
        typer.Option("--gold-root", help="Explicit gold root."),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", help="Synthetic split to evaluate."),
    ] = None,
    allow_final_test_evaluation: Annotated[
        bool,
        typer.Option(
            "--allow-final-test-evaluation",
            help="Allow protected synthetic test evaluation.",
        ),
    ] = False,
) -> None:
    """Evaluate parser output against manual or synthetic gold annotations."""
    if gold_root is not None:
        actual_split = split or "development"
        try:
            report = evaluate_synthetic_cv_extraction(
                root=gold_root,
                split=actual_split,
                allow_final_test_evaluation=allow_final_test_evaluation,
            )
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo("Synthetic CV extraction validation")
        typer.echo(f"split: {actual_split}")
        typer.echo(f"json: {gold_root / 'reports' / (actual_split + '_evaluation.json')}")
        typer.echo(f"markdown: {gold_root / 'reports' / (actual_split + '_evaluation.md')}")
        typer.echo(f"parser_success_rate: {report['metrics']['parser_success_rate']}")
        return
    report = evaluate_cv_extraction()
    gates = report["quality_gates"]
    typer.echo("CV extraction validation")
    typer.echo("json: datasets/reports/cv_extraction_validation.json")
    typer.echo("markdown: datasets/reports/cv_extraction_validation.md")
    typer.echo(f"quality_gates_passed: {gates['passed']}")
    if not gates["passed"]:
        raise typer.Exit(code=1)


@skills_app.command("sync")
def skills_sync_cmd() -> None:
    """Upsert the skills ontology YAML into the skills table."""
    result = asyncio.run(sync_skills())
    typer.echo(f"Synced {result['synced']} skills.")


@enrichment_app.command("run-matcher")
def enrichment_run_matcher_cmd(
    source: str | None = typer.Option(None, "--source", help="Only process jobs from this source."),
    country: str | None = typer.Option(
        None,
        "--country",
        help="Only process jobs in this country.",
    ),
    min_job_id: int | None = typer.Option(None, "--min-job-id", help="Minimum job ID."),
    max_job_id: int | None = typer.Option(None, "--max-job-id", help="Maximum job ID."),
    limit: int | None = typer.Option(None, "--limit", help="Maximum jobs to process."),
    batch_size: int = typer.Option(500, "--batch-size", help="Jobs per committed batch."),
    resume_run_id: int | None = typer.Option(
        None,
        "--resume-run-id",
        help="Resume an existing run.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Create a fresh run even if jobs were processed.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm large or unlimited runs."),
) -> None:
    """Run deterministic matcher enrichment."""
    if resume_run_id is not None and force:
        raise typer.BadParameter("--resume-run-id cannot be used with --force")

    config = MatcherEnrichmentConfig(
        source=source,
        country=country,
        min_job_id=min_job_id,
        max_job_id=max_job_id,
        limit=limit,
        batch_size=batch_size,
    )
    _confirm_enrichment_scope(config, yes=yes)

    ontology = load_ontology()
    versions = get_matcher_versions(ontology)
    typer.echo("Matcher enrichment")
    typer.echo(f"matcher_version: {versions.matcher_version}")
    typer.echo(f"ontology_version: {versions.ontology_version}")
    typer.echo(f"config: {config.to_run_config()}")

    started = perf_counter()
    try:
        summary, report = asyncio.run(
            _run_matcher_and_collect_report(
                config,
                resume_run_id=resume_run_id,
                force=force,
            )
        )
    except (EnrichmentConfigError, OntologySyncError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    elapsed = perf_counter() - started

    typer.echo(f"run_id: {summary.run_id}")
    typer.echo(f"status: {summary.status}")
    typer.echo(f"jobs_total: {summary.jobs_total}")
    typer.echo(f"jobs_processed: {summary.jobs_processed}")
    typer.echo(f"jobs_succeeded: {summary.jobs_succeeded}")
    typer.echo(f"jobs_failed: {summary.jobs_failed}")
    typer.echo(f"jobs_with_skills: {summary.jobs_with_skills}")
    typer.echo(f"zero_skill_jobs: {summary.zero_skill_jobs}")
    typer.echo(f"persisted_skill_rows: {report.persisted_skill_rows}")
    typer.echo(f"elapsed_seconds: {elapsed:.2f}")


@enrichment_app.command("report")
def enrichment_report_cmd(
    run_id: int = typer.Argument(..., help="Enrichment run ID to report."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    top_n: int = typer.Option(20, "--top-n", help="Number of top skills to include."),
) -> None:
    """Show a persisted enrichment run report."""
    try:
        report = asyncio.run(collect_enrichment_report(run_id, top_n=top_n))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(enrichment_report_to_json(report))
    else:
        typer.echo(format_enrichment_report(report))


@llm_app.command("benchmark")
def llm_benchmark_cmd(
    run_id: int = typer.Option(113, "--run-id", help="Matcher enrichment run to sample."),
    sample_size: int = typer.Option(100, "--sample-size", help="Deterministic sample size."),
    output_dir: str = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir", help="Report/cache directory."
    ),
    model: str | None = typer.Option(None, "--model", help="Override LLM model."),
    max_jobs: int | None = typer.Option(
        None, "--max-jobs", help="Limit candidate jobs before sampling."
    ),
    resume: bool = typer.Option(False, "--resume", help="Reuse cached responses where present."),
    offline_report: bool = typer.Option(
        False,
        "--offline-report",
        help="Rebuild reports from cached responses without calling Groq.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Do not call Groq; write empty cache/report."
    ),
) -> None:
    """Benchmark optional guarded Groq extraction without production writes."""
    try:
        result = asyncio.run(
            run_groq_benchmark(
                run_id=run_id,
                sample_size=sample_size,
                output_dir=Path(output_dir),
                model=model,
                max_jobs=max_jobs,
                resume=resume,
                offline_report=offline_report,
                dry_run=dry_run,
            )
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    metrics = result.metrics
    typer.echo("Groq benchmark")
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"sample_size: {len(result.records)}")
    typer.echo(f"total_groq_calls: {metrics.total_groq_calls}")
    typer.echo(f"valid_json_rate: {metrics.valid_json_rate:.1%}")
    typer.echo(f"guard_acceptance_rate: {metrics.guard_acceptance_rate:.1%}")
    typer.echo(f"guard_rejection_rate: {metrics.guard_rejection_rate:.1%}")
    typer.echo(
        f"mean_accepted_skills_added_per_job: {metrics.mean_accepted_skills_added_per_job:.2f}"
    )
    typer.echo(f"estimated_total_cost_usd: {metrics.estimated_total_cost_usd:.6f}")


@embed_app.command("jobs")
def embed_jobs_cmd(
    limit: int | None = typer.Option(None, "--limit", help="Maximum jobs to embed."),
    batch_size: int = typer.Option(64, "--batch-size", help="Jobs per embedding batch."),
    re_embed: bool = typer.Option(False, "--re-embed", help="Recompute current embeddings."),
    job_id: int | None = typer.Option(None, "--job-id", help="Embed one job ID."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Select and report without encoding."),
    model: str | None = typer.Option(None, "--model", help="Override embedding model."),
    device: str = typer.Option("cpu", "--device", help="cpu or cuda."),
    run_id: int = typer.Option(342, "--run-id", help="Enrichment run for canonical skills."),
) -> None:
    """Embed normalized jobs for semantic retrieval."""
    if device not in {"cpu", "cuda"}:
        raise typer.BadParameter("--device must be cpu or cuda")
    config = JobEmbeddingConfig(
        limit=limit,
        batch_size=batch_size,
        re_embed=re_embed,
        job_id=job_id,
        dry_run=dry_run,
        model=model,
        device=device,
        run_id=run_id,
    )
    try:
        summary = asyncio.run(run_job_embedding(config))
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Job embeddings")
    typer.echo(f"model: {summary.model_name}")
    typer.echo(f"embedding_version: {summary.embedding_version}")
    typer.echo(f"dimension: {summary.dimension}")
    typer.echo(f"selected_jobs: {summary.selected_jobs}")
    typer.echo(f"embedded_jobs: {summary.embedded_jobs}")
    typer.echo(f"pending_jobs: {summary.pending_jobs}")
    typer.echo(f"failed_batches: {summary.failed_batches}")
    typer.echo(f"dry_run: {summary.dry_run}")
    typer.echo(f"elapsed_seconds: {summary.elapsed_seconds:.2f}")
    typer.echo(f"average_jobs_per_second: {summary.average_jobs_per_second:.2f}")



@app.command("recommend-from-cv")
def recommend_from_cv_cmd(
    path: str | None = typer.Argument(None, help="Path to a TXT, PDF, or DOCX CV."),
    cv_path: str | None = typer.Option(None, "--cv", help="Path to a TXT, PDF, or DOCX CV."),
    limit: int = typer.Option(10, "--limit", help="Number of ranked jobs to return."),
    run_id: int = typer.Option(
        DEFAULT_RECOMMENDATION_RUN_ID,
        "--run-id",
        help="Single enrichment run ID to use for job skills.",
    ),
    mode: str = typer.Option("lexical", "--mode", help="lexical, semantic, or hybrid."),
    alpha: float = typer.Option(0.60, "--alpha", help="Hybrid lexical weight in 0..1."),
    country: str | None = typer.Option(None, "--country", help="Preferred job country filter."),
    city: str | None = typer.Option(None, "--city", help="Preferred job city filter."),
    contract_type: str | None = typer.Option(
        None,
        "--contract-type",
        help="Optional contract type filter applied to returned recommendations.",
    ),
    remote: bool | None = typer.Option(None, "--remote/--no-remote", help="Remote preference."),
    confirmed_profile: Annotated[
        Path | None,
        typer.Option(
            "--confirmed-profile",
            help="Optional JSON file containing a user-confirmed profile for matching.",
        ),
    ] = None,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run the complete privacy-safe CV-to-job recommendation workflow."""
    _run_cv_recommendation_cli(
        path=_resolve_cv_path(path, cv_path),
        limit=limit,
        run_id=run_id,
        mode=mode,
        alpha=alpha,
        country=country,
        city=city,
        contract_type=contract_type,
        remote=remote,
        confirmed_profile=confirmed_profile,
        json_output=json_output,
    )


@cv_app.command("recommend")
def cv_recommend_cmd(
    path: str | None = typer.Argument(None, help="Path to a TXT, PDF, or DOCX CV."),
    cv_path: str | None = typer.Option(None, "--cv", help="Path to a TXT, PDF, or DOCX CV."),
    limit: int = typer.Option(10, "--limit", help="Number of ranked jobs to return."),
    run_id: int = typer.Option(
        DEFAULT_RECOMMENDATION_RUN_ID,
        "--run-id",
        help="Single enrichment run ID to use for job skills.",
    ),
    mode: str = typer.Option("lexical", "--mode", help="lexical, semantic, or hybrid."),
    alpha: float = typer.Option(0.60, "--alpha", help="Hybrid lexical weight in 0..1."),
    country: str | None = typer.Option(None, "--country", help="Preferred job country filter."),
    city: str | None = typer.Option(None, "--city", help="Preferred job city filter."),
    contract_type: str | None = typer.Option(
        None,
        "--contract-type",
        help="Optional contract type filter applied to returned recommendations.",
    ),
    remote: bool | None = typer.Option(None, "--remote/--no-remote", help="Remote preference."),
    confirmed_profile: Annotated[
        Path | None,
        typer.Option(
            "--confirmed-profile",
            help="Optional JSON file containing a user-confirmed profile for matching.",
        ),
    ] = None,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run the complete CV recommendation workflow from the CV command group."""
    _run_cv_recommendation_cli(
        path=_resolve_cv_path(path, cv_path),
        limit=limit,
        run_id=run_id,
        mode=mode,
        alpha=alpha,
        country=country,
        city=city,
        contract_type=contract_type,
        remote=remote,
        confirmed_profile=confirmed_profile,
        json_output=json_output,
    )


def _resolve_cv_path(path: str | None, cv_path: str | None) -> str:
    if path and cv_path:
        raise typer.BadParameter("provide either positional CV path or --cv, not both")
    value = path or cv_path
    if not value:
        raise typer.BadParameter("a CV path is required")
    return value


def _run_cv_recommendation_cli(
    *,
    path: str,
    limit: int,
    run_id: int,
    mode: str,
    alpha: float,
    country: str | None,
    city: str | None,
    contract_type: str | None,
    remote: bool | None,
    confirmed_profile: Path | None,
    json_output: bool,
) -> None:
    if limit <= 0:
        raise typer.BadParameter("--limit must be positive")
    confirmed = _load_confirmed_profile(confirmed_profile)
    request = CvRecommendationRequest(
        cv_path=path,
        limit=limit,
        run_id=run_id,
        mode=cast(RecommendationMode, mode),
        alpha=alpha,
        filters=RecommendationFilters(
            country=country,
            city=city,
            contract_type=contract_type,
            remote_preference=remote,
        ),
        confirmed_profile=confirmed,
    )
    try:
        result = asyncio.run(run_cv_recommendation_workflow(request))
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(workflow_response_to_json(result))
    else:
        typer.echo(format_workflow_response(result))


def _load_confirmed_profile(path: Path | None) -> ConfirmedProfileInput | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ConfirmedProfileInput.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid confirmed profile JSON: {exc}") from exc

@cv_app.command("match")
def cv_match_cmd(
    path: str = typer.Argument(..., help="Path to a TXT, PDF, or DOCX CV."),
    top: int = typer.Option(20, "--top", help="Number of ranked jobs to return."),
    run_id: int = typer.Option(
        DEFAULT_RECOMMENDATION_RUN_ID,
        "--run-id",
        help="Single enrichment run ID to use for job skills.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    deterministic_only: bool = typer.Option(
        True,
        "--deterministic-only/--allow-guarded-candidates",
        help=(
            "Use deterministic CV skills only unless offline guarded candidates are supplied later."
        ),
    ),
    mode: str = typer.Option("lexical", "--mode", help="lexical, semantic, or hybrid."),
    alpha: float = typer.Option(0.60, "--alpha", help="Hybrid lexical weight in 0..1."),
) -> None:
    """Parse a CV and rank jobs using deterministic skills."""
    if top <= 0:
        raise typer.BadParameter("--top must be positive")
    if not deterministic_only:
        typer.echo(
            "Error: guarded candidate input is not exposed in this CLI phase; "
            "use --deterministic-only.",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        result = asyncio.run(
            match_cv_document(
                path, top=top, run_id=run_id, mode=cast(RecommendationMode, mode), alpha=alpha
            )
        )
    except (CvDocumentError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(cv_match_result_to_json(result))
    else:
        typer.echo(format_cv_match_result(result))


async def _run_matcher_and_collect_report(
    config: MatcherEnrichmentConfig,
    *,
    resume_run_id: int | None,
    force: bool,
) -> tuple[MatcherEnrichmentSummary, EnrichmentRunReport]:
    summary = await run_matcher_enrichment(
        config,
        resume_run_id=resume_run_id,
        force=force,
    )
    report = await collect_enrichment_report(summary.run_id)
    return summary, report


def _confirm_enrichment_scope(config: MatcherEnrichmentConfig, *, yes: bool) -> None:
    if config.limit is not None and config.limit <= 1000:
        return
    if yes:
        return
    scope = "unlimited" if config.limit is None else f"{config.limit:,} jobs"
    confirmed = typer.confirm(f"This enrichment run may process {scope}. Continue?")
    if not confirmed:
        raise typer.Abort()


if __name__ == "__main__":
    app()
