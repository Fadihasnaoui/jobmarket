import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3, Briefcase, Building2, Radar, Sparkles } from "lucide-react";
import { ApiError, getStatsOverview, getStatsSkills } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { ErrorState, LoadingState } from "../components/StatusStates";
import type { StatsOverviewResponse, StatsSkillsResponse } from "../api/types";

const COUNTRIES = ["", "fr", "de", "gb", "es", "nl", "it"];
const PIE_COLORS = ["#6366f1", "#a855f7", "#d946ef", "#06b6d4", "#10b981", "#f59e0b"];

const TOOLTIP_STYLE = {
  borderRadius: 12,
  border: "1px solid #e2e8f0",
  boxShadow: "0 4px 16px -4px rgb(15 23 42 / 0.12)",
  fontSize: 13,
};

export function DashboardPage() {
  const [overview, setOverview] = useState<StatsOverviewResponse | null>(null);
  const [topSkills, setTopSkills] = useState<StatsSkillsResponse | null>(null);
  const [countrySkills, setCountrySkills] = useState<StatsSkillsResponse | null>(null);
  const [country, setCountry] = useState("fr");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = () => {
    setIsLoading(true);
    setError(null);
    Promise.all([getStatsOverview(), getStatsSkills({ limit: 10 })])
      .then(([overviewData, skillsData]) => {
        setOverview(overviewData);
        setTopSkills(skillsData);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load stats."))
      .finally(() => setIsLoading(false));
  };

  useEffect(fetchAll, []);

  useEffect(() => {
    if (!country) {
      setCountrySkills(null);
      return;
    }
    getStatsSkills({ limit: 10, country }).then(setCountrySkills).catch(() => setCountrySkills(null));
  }, [country]);

  if (isLoading) return <LoadingState label="Loading corpus statistics…" />;
  if (error) return <ErrorState message={error} onRetry={fetchAll} />;
  if (!overview || !topSkills) return null;

  return (
    <div className="space-y-8">
      <PageHeader
        icon={BarChart3}
        title="Corpus dashboard"
        subtitle={`Reference run ${overview.reference_run_id}.`}
      />

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard icon={Briefcase} label="Total jobs" value={overview.jobs.toLocaleString()} />
        <StatCard icon={Building2} label="Companies" value={overview.companies.toLocaleString()} />
        <StatCard
          icon={Sparkles}
          label="Skill coverage"
          value={`${overview.skill_coverage_pct.toFixed(1)}%`}
          sub={`${overview.jobs_with_skills.toLocaleString()} / ${overview.jobs_in_reference_run.toLocaleString()}`}
        />
        <StatCard
          icon={Radar}
          label="Embedding coverage"
          value={`${overview.embedding_coverage_pct.toFixed(1)}%`}
          sub={`${overview.embedded_jobs.toLocaleString()} jobs`}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Top skills (overall demand)">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={topSkills.skills} layout="vertical" margin={{ left: 24 }}>
              <defs>
                <linearGradient id="bar-gradient-indigo" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#818cf8" />
                  <stop offset="100%" stopColor="#4f46e5" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="canonical_skill"
                width={140}
                tick={{ fontSize: 12, fill: "#475569" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "#f8fafc" }}
                contentStyle={TOOLTIP_STYLE}
                formatter={(value: number) => [`${value} jobs`, "demand"]}
              />
              <Bar dataKey="job_count" fill="url(#bar-gradient-indigo)" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Top skills by country"
          action={
            <select
              value={country}
              onChange={(e) => setCountry(e.target.value)}
              className="rounded-full border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600"
            >
              {COUNTRIES.filter(Boolean).map((c) => (
                <option key={c} value={c}>
                  {c.toUpperCase()}
                </option>
              ))}
            </select>
          }
        >
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={countrySkills?.skills ?? []} layout="vertical" margin={{ left: 24 }}>
              <defs>
                <linearGradient id="bar-gradient-fuchsia" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#e879f9" />
                  <stop offset="100%" stopColor="#a855f7" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="canonical_skill"
                width={140}
                tick={{ fontSize: 12, fill: "#475569" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "#f8fafc" }}
                contentStyle={TOOLTIP_STYLE}
                formatter={(value: number) => [`${value} jobs`, "demand"]}
              />
              <Bar dataKey="job_count" fill="url(#bar-gradient-fuchsia)" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <ChartCard title="Source breakdown (raw ingested rows)">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie
              data={overview.per_source}
              dataKey="raw_total"
              nameKey="source"
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={90}
              paddingAngle={2}
              label={(entry) => `${entry.source}: ${entry.raw_total.toLocaleString()}`}
            >
              {overview.per_source.map((entry, index) => (
                <Cell key={entry.source} fill={PIE_COLORS[index % PIE_COLORS.length]} stroke="white" strokeWidth={2} />
              ))}
            </Pie>
            <Tooltip contentStyle={TOOLTIP_STYLE} />
          </PieChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: typeof Briefcase;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-4 shadow-card transition-shadow hover:shadow-soft">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
          <Icon className="h-4 w-4" strokeWidth={2} />
        </div>
        <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      </div>
      <div className="font-display mt-2 text-2xl font-bold text-slate-900">{value}</div>
      {sub && <div className="text-xs text-slate-400">{sub}</div>}
    </div>
  );
}

function ChartCard({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-4 shadow-card">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-display text-sm font-semibold text-slate-700">{title}</h3>
        {action}
      </div>
      {children}
    </div>
  );
}
