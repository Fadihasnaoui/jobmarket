import { useEffect, useState } from "react";
import { TrendingUp } from "lucide-react";
import { ApiError, getSkillGap } from "../api/client";
import { HonestyBanners } from "../components/HonestyBanners";
import { PageHeader } from "../components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/StatusStates";
import { useCvSession } from "../context/CvSessionContext";
import type { SkillGapResponse } from "../api/types";

export function SkillGapPage() {
  const { cvId } = useCvSession();
  const [data, setData] = useState<SkillGapResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGap = () => {
    if (!cvId) return;
    setIsLoading(true);
    setError(null);
    getSkillGap(cvId, { top: 20 })
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load skill gap."))
      .finally(() => setIsLoading(false));
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(fetchGap, [cvId]);

  if (!cvId) {
    return (
      <EmptyState>
        Upload a CV first on the <strong>Upload</strong> tab to see your skill gap.
      </EmptyState>
    );
  }

  const maxDemand = data ? Math.max(1, ...data.gaps.map((g) => g.overall_job_demand)) : 1;

  return (
    <div className="space-y-6">
      <PageHeader
        icon={TrendingUp}
        title="Skill gap"
        subtitle={`Skills missing across your top ${data?.top_matches_considered ?? "…"} matches, ranked by how many jobs in the corpus require them.`}
      />

      {isLoading && <LoadingState label="Aggregating missing skills across top matches…" />}
      {error && <ErrorState message={error} onRetry={fetchGap} />}

      {data && !isLoading && (
        <>
          <HonestyBanners
            outOfScope={data.out_of_scope}
            outOfScopeMessage={data.message}
            warnings={data.warnings}
            message={data.message}
          />

          {data.gaps.length === 0 ? (
            <EmptyState>
              {data.out_of_scope
                ? "No skill gap analysis — this profile is out of scope for this platform."
                : "No missing skills detected across your top matches — nice CV."}
            </EmptyState>
          ) : (
            <ul className="space-y-2.5">
              {data.gaps.map((gap, index) => (
                <li
                  key={gap.canonical_skill}
                  className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white px-4 py-3.5 shadow-card transition-shadow hover:shadow-soft"
                  style={{ animationDelay: `${Math.min(index, 10) * 0.04}s` }}
                >
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <div className="flex items-center gap-3">
                      <RankBadge rank={index + 1} />
                      <span className="font-semibold text-slate-800">{gap.canonical_skill}</span>
                      {gap.category && (
                        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                          {gap.category}
                        </span>
                      )}
                    </div>
                    <div className="shrink-0 text-right text-xs text-slate-500">
                      missing in <strong className="text-slate-700">{gap.missing_in_top_matches}</strong>{" "}
                      of your top matches
                      <span className="mx-2 text-slate-300">|</span>
                      <strong className="text-slate-700">
                        {gap.overall_job_demand.toLocaleString()}
                      </strong>{" "}
                      jobs corpus-wide
                    </div>
                  </div>
                  <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-1.5 rounded-full bg-gradient-to-r from-indigo-500 to-fuchsia-500 transition-all"
                      style={{ width: `${(gap.overall_job_demand / maxDemand) * 100}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

function RankBadge({ rank }: { rank: number }) {
  const isTopThree = rank <= 3;
  return (
    <span
      className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-xs font-bold ${
        isTopThree
          ? "bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white"
          : "bg-slate-100 text-slate-500"
      }`}
    >
      {rank}
    </span>
  );
}
