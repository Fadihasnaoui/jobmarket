import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  MinusCircle,
  XCircle,
} from "lucide-react";
import { ApiError, getQualityReport } from "../api/client";
import { HonestyBanners } from "../components/HonestyBanners";
import { PageHeader } from "../components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/StatusStates";
import { useCvSession } from "../context/CvSessionContext";
import type {
  QualityCheck,
  QualityCheckStatus,
  QualityReportResponse,
  RecommendationMode,
} from "../api/types";

const MODES: RecommendationMode[] = ["hybrid", "lexical", "semantic"];
export function QualityReportPage() {
  const { cvId } = useCvSession();
  const [mode, setMode] = useState<RecommendationMode>("hybrid");
  const [data, setData] = useState<QualityReportResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCheckId, setSelectedCheckId] = useState<string | null>(null);
  // Guards against the auto-fetch-on-mount request and a manual "Re-check" click
  // resolving out of order — only the response from the most recently fired
  // request is ever allowed to update state.
  const requestIdRef = useRef(0);

  const fetchReport = () => {
    if (!cvId) return;
    const requestId = ++requestIdRef.current;
    setIsLoading(true);
    setError(null);
    getQualityReport(
      cvId,
      {},
      { mode },
    )
      .then((result) => {
        if (requestIdRef.current !== requestId) return;
        setData(result);
        setSelectedCheckId((current) => current ?? result.checks[0]?.id ?? null);
      })
      .catch((err) => {
        if (requestIdRef.current !== requestId) return;
        setError(err instanceof ApiError ? err.message : "Failed to load report.");
      })
      .finally(() => {
        if (requestIdRef.current === requestId) setIsLoading(false);
      });
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(fetchReport, [cvId, mode]);

  if (!cvId) {
    return (
      <EmptyState>
        Upload a CV first on the <strong>Upload</strong> tab to see its quality report.
      </EmptyState>
    );
  }

  const selectedCheck = data?.checks.find((c) => c.id === selectedCheckId) ?? null;

  return (
    <div className="space-y-6">
      <PageHeader
        icon={ClipboardCheck}
        title="CV Quality Report"
        subtitle="A scored diagnostic of your real, extracted content — nothing here is invented."
        action={
          <div className="flex gap-1 rounded-full bg-slate-100 p-1">
            {MODES.map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`rounded-full px-3.5 py-1.5 text-xs font-semibold capitalize transition-all ${
                  mode === m
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        }
      />

      {isLoading && !data && <LoadingState label="Scoring your CV against six checks…" />}
      {error && <ErrorState message={error} onRetry={fetchReport} />}

      {data && (
        <>
          <HonestyBanners
            outOfScope={data.out_of_scope}
            outOfScopeMessage={data.message}
            extractionDegraded={data.extraction_degraded || data.source_degraded}
            extractionDegradedReason={data.source_degraded_reason}
            lowConfidence={data.low_confidence}
            warnings={data.warnings}
            message={data.message}
          />

          {!data.out_of_scope && (
            <>
              <div className="flex flex-col gap-6 lg:flex-row">
                <div className="flex shrink-0 flex-col gap-4 lg:w-72">
                  <ScorePanel score={data.overall_score} />
                  <CategoryList
                    report={data}
                    selectedCheckId={selectedCheckId}
                    onSelect={setSelectedCheckId}
                  />
                </div>
                <div className="min-w-0 flex-1">
                  {selectedCheck ? (
                    <CategoryDetail check={selectedCheck} report={data} />
                  ) : (
                    <EmptyState>No checks available for this CV.</EmptyState>
                  )}
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

const STATUS_STYLES: Record<QualityCheckStatus, { label: string; className: string }> = {
  pass: { label: "Pass", className: "bg-emerald-100 text-emerald-700" },
  warning: { label: "Warning", className: "bg-amber-100 text-amber-700" },
  fail: { label: "Needs work", className: "bg-rose-100 text-rose-700" },
  not_checked: { label: "Not checked", className: "bg-slate-100 text-slate-600" },
};

function ScorePanel({ score }: { score: number }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 text-center shadow-card">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Overall score</p>
      <p className="mt-2 font-display text-5xl font-bold text-slate-900">{Math.round(score)}</p>
      <p className="mt-1 text-sm text-slate-500">out of 100</p>
    </div>
  );
}

function CategoryList({ report, selectedCheckId, onSelect }: { report: QualityReportResponse; selectedCheckId: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-2 shadow-card">
      {report.checks.map((check) => {
        const style = STATUS_STYLES[check.status];
        return (
          <button key={check.id} onClick={() => onSelect(check.id)} className={`flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left text-sm transition-colors ${selectedCheckId === check.id ? "bg-violet-50 text-violet-800" : "hover:bg-slate-50"}`}>
            <span className="font-medium">{check.label}</span>
            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${style.className}`}>{style.label}</span>
          </button>
        );
      })}
    </div>
  );
}

function CategoryDetail({ check, report }: { check: QualityCheck; report: QualityReportResponse }) {
  const style = STATUS_STYLES[check.status];
  const Icon = check.status === "pass" ? CheckCircle2 : check.status === "warning" ? AlertTriangle : check.status === "fail" ? XCircle : MinusCircle;
  return (
    <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-card">
      <div className="flex items-start justify-between gap-3">
        <div><h2 className="font-display text-xl font-semibold text-slate-900">{check.label}</h2><p className="mt-1 text-sm text-slate-500">{check.score} / {check.max_score} points</p></div>
        <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${style.className}`}><Icon className="h-3.5 w-3.5" />{style.label}</span>
      </div>
      <p className="leading-relaxed text-slate-700">{check.message}</p>
      {check.suggestion && <div className="rounded-xl bg-violet-50 p-4 text-sm text-violet-900"><strong>Suggestion:</strong> {check.suggestion}</div>}
      {check.id === "spelling_grammar" && report.spelling_issues.length > 0 && (
        <DetailList title="Spelling suggestions" items={report.spelling_issues.map((item) => `${item.original} to ${item.corrected}`)} />
      )}
      {check.id === "impact_quantification" && report.unquantified_bullets.length > 0 && (
        <DetailList title="Experience bullets that could use a measurable result" items={report.unquantified_bullets.map((item) => item.text)} />
      )}
      {check.id === "market_skill_gap" && report.recommended_skills_to_develop.length > 0 && (
        <DetailList title="Skills to develop" items={report.recommended_skills_to_develop.map((item) => item.canonical_skill)} />
      )}
    </div>
  );
}

function DetailList({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-slate-800">{title}</h3>
      <ul className="space-y-2 text-sm text-slate-600">
        {items.slice(0, 8).map((item, index) => (
          <li key={`${item}-${index}`} className="leading-relaxed">- {item}</li>
        ))}
      </ul>
    </div>
  );
}
