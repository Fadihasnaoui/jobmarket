import { useEffect, useState } from "react";
import {
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  Info,
  MessageCircle,
  Sparkles,
  Target,
  XCircle,
} from "lucide-react";
import { ApiError, getMatches } from "../api/client";
import { HonestyBanners } from "../components/HonestyBanners";
import { PageHeader } from "../components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/StatusStates";
import { useCvSession } from "../context/CvSessionContext";
import type { ChatJobReference, MatchesResponse, RecommendationMode, RecommendationOutput } from "../api/types";

const MODES: RecommendationMode[] = ["hybrid", "lexical", "semantic"];

export function MatchesPage({
  onMatchesLoaded,
  onAskAboutJob,
}: {
  onMatchesLoaded: (jobs: ChatJobReference[]) => void;
  onAskAboutJob: (job: ChatJobReference) => void;
}) {
  const { cvId } = useCvSession();
  const [mode, setMode] = useState<RecommendationMode>("hybrid");
  const [data, setData] = useState<MatchesResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMatches = () => {
    if (!cvId) return;
    setIsLoading(true);
    setError(null);
    getMatches(cvId, { mode, limit: 10 })
      .then((result) => {
        setData(result);
        onMatchesLoaded(
          result.matches.map((match) => ({
            jobId: match.job_id,
            title: match.title,
            company: match.company,
          })),
        );
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load matches."))
      .finally(() => setIsLoading(false));
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(fetchMatches, [cvId, mode]);

  if (!cvId) {
    return (
      <EmptyState>
        Upload a CV first on the <strong>Upload</strong> tab to see job matches.
      </EmptyState>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Target}
        title="Job matches"
        subtitle={`Ranked against the full corpus (run ${data?.run_id ?? "…"}).`}
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

      <p className="flex items-center gap-1.5 text-xs text-slate-400">
        <Info className="h-3.5 w-3.5 shrink-0" strokeWidth={2} />
        Jobs are drawn from a snapshot of the market; some postings may have expired
        since they were collected.
      </p>

      {isLoading && <LoadingState label="Ranking jobs against the corpus…" />}
      {error && <ErrorState message={error} onRetry={fetchMatches} />}

      {data && !isLoading && (
        <>
          <HonestyBanners
            outOfScope={data.out_of_scope}
            outOfScopeMessage={data.message}
            extractionDegraded={data.extraction_degraded}
            lowConfidence={data.low_confidence}
            warnings={data.warnings}
            message={data.message}
          />

          {data.matches.length === 0 ? (
            <EmptyState>
              {data.out_of_scope
                ? "No recommendations — this profile is out of scope for this platform."
                : "No matching jobs found for this CV and filter combination."}
            </EmptyState>
          ) : (
            <ul className="space-y-3">
              {data.matches.map((match, index) => (
                <MatchCard key={match.job_id} match={match} index={index} onAskAboutJob={onAskAboutJob} />
              ))}
            </ul>
          )}

        </>
      )}
    </div>
  );
}

function scoreTier(score: number): { ring: string; text: string } {
  if (score >= 75) return { ring: "ring-emerald-200 bg-emerald-50", text: "text-emerald-700" };
  if (score >= 50) return { ring: "ring-amber-200 bg-amber-50", text: "text-amber-700" };
  return { ring: "ring-slate-200 bg-slate-50", text: "text-slate-600" };
}

const STALE_DAYS_THRESHOLD = 120;

function describePosting(postedAt: string | null): { label: string; stale: boolean } | null {
  if (!postedAt) return null;
  const posted = new Date(postedAt);
  if (Number.isNaN(posted.getTime())) return null;
  const days = Math.floor((Date.now() - posted.getTime()) / (1000 * 60 * 60 * 24));
  const formatted = posted.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  const relative =
    days <= 0
      ? "today"
      : days === 1
        ? "1 day ago"
        : days < 60
          ? `${days} days ago`
          : days < 730
            ? `${Math.round(days / 30)} months ago`
            : `${Math.round(days / 365)} years ago`;
  return { label: `Posted ${formatted} (${relative})`, stale: days >= STALE_DAYS_THRESHOLD };
}

function MatchCard({
  match,
  index,
  onAskAboutJob,
}: {
  match: RecommendationOutput;
  index: number;
  onAskAboutJob: (job: ChatJobReference) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const isSemanticSourced = match.retrieval_source === "semantic" || match.retrieval_source === "both";
  const tier = scoreTier(match.final_score);
  const posting = describePosting(match.posted_at);

  return (
    <li
      className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-5 shadow-card transition-shadow hover:shadow-soft"
      style={{ animationDelay: `${Math.min(index, 8) * 0.05}s` }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-display font-semibold text-slate-900">{match.title}</h3>
            {isSemanticSourced && (
              <span
                title="This job was found via semantic similarity, not just keyword overlap."
                className="inline-flex items-center gap-1 rounded-full bg-gradient-to-r from-violet-100 to-fuchsia-100 px-2.5 py-0.5 text-xs font-semibold text-fuchsia-700 ring-1 ring-fuchsia-200"
              >
                <Sparkles className="h-3 w-3" strokeWidth={2.5} />
                semantic discovery
              </span>
            )}
          </div>
          <p className="mt-0.5 text-sm text-slate-500">
            {match.company}
            {match.location && ` · ${match.location}`}
            {match.contract_type && ` · ${match.contract_type}`}
          </p>
          {posting && (
            <p
              className={`mt-1 flex items-center gap-1.5 text-xs ${
                posting.stale ? "text-amber-600" : "text-slate-400"
              }`}
            >
              <CalendarClock className="h-3 w-3 shrink-0" strokeWidth={2} />
              {posting.label}
              {posting.stale && (
                <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
                  older posting
                </span>
              )}
            </p>
          )}
        </div>
        <div
          className={`flex h-16 w-16 shrink-0 flex-col items-center justify-center rounded-2xl ring-2 ${tier.ring}`}
        >
          <div className={`font-display text-xl font-bold ${tier.text}`}>
            {match.final_score.toFixed(0)}
          </div>
          <div className="text-[10px] uppercase tracking-wide text-slate-400">score</div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-3 text-xs">
        <SkillGroup label="Matched" icon={CheckCircle2} items={match.matched_skills} tone="emerald" />
        <SkillGroup label="Missing" icon={XCircle} items={match.missing_important_skills} tone="red" />
      </div>

      <p className="mt-3 text-sm leading-relaxed text-slate-600">{match.explanation}</p>

      <div className="mt-3 flex flex-wrap items-center gap-4">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-1 text-xs font-semibold text-indigo-600 transition-colors hover:text-indigo-800"
        >
          <ChevronDown
            className={["h-3.5 w-3.5 transition-transform", expanded ? "rotate-180" : ""].join(" ")}
            strokeWidth={2.5}
          />
          {expanded ? "hide details" : "why this score"}
        </button>
        <button
          onClick={() => onAskAboutJob({ jobId: match.job_id, title: match.title, company: match.company })}
          className="flex items-center gap-1 text-xs font-semibold text-violet-600 transition-colors hover:text-violet-800"
        >
          <MessageCircle className="h-3.5 w-3.5" strokeWidth={2.25} />
          Ask AI about this job
        </button>
      </div>

      {expanded && (
        <div className="mt-3 grid animate-fade-in grid-cols-2 gap-3 rounded-xl bg-slate-50 p-4 text-sm sm:grid-cols-4">
          <ScoreStat label="Final" value={match.final_score.toFixed(2)} />
          <ScoreStat label="Lexical" value={formatScore(match.lexical_score)} />
          <ScoreStat label="Semantic" value={formatScore(match.semantic_score)} />
          <ScoreStat label="Retrieval source" value={match.retrieval_source} />
          <ScoreStat label="Skill score" value={match.skill_score.toFixed(2)} />
          <ScoreStat label="Role/domain" value={match.role_domain_score.toFixed(2)} />
          <ScoreStat label="Career fit" value={match.career_level_compatibility.toFixed(2)} />
          {match.source_url && (
            <a
              href={match.source_url}
              target="_blank"
              rel="noreferrer"
              className="col-span-2 inline-flex items-center gap-1 font-medium text-indigo-600 hover:text-indigo-800 sm:col-span-4"
            >
              View original posting <ExternalLink className="h-3.5 w-3.5" strokeWidth={2} />
            </a>
          )}
        </div>
      )}
    </li>
  );
}

function formatScore(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(2);
}

function ScoreStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="font-mono font-semibold text-slate-800">{value}</div>
    </div>
  );
}

function SkillGroup({
  label,
  icon: Icon,
  items,
  tone,
}: {
  label: string;
  icon: typeof CheckCircle2;
  items: string[];
  tone: "emerald" | "red";
}) {
  if (items.length === 0) return null;
  const toneClasses = tone === "emerald" ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700";
  const iconClasses = tone === "emerald" ? "text-emerald-500" : "text-red-400";
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="flex items-center gap-1 font-semibold text-slate-400">
        <Icon className={`h-3.5 w-3.5 ${iconClasses}`} strokeWidth={2.25} />
        {label}:
      </span>
      {items.map((item) => (
        <span key={item} className={`rounded-md px-1.5 py-0.5 ${toneClasses}`}>
          {item}
        </span>
      ))}
    </div>
  );
}
