import { useEffect, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  Database,
  FileSearch,
  Sparkles,
  Target,
  Upload,
} from "lucide-react";
import { getStatsOverview } from "../api/client";
import { AuroraBackground, MatchGraphIllustration } from "../components/illustrations";
import type { StatsOverviewResponse } from "../api/types";

export function LandingPage({ onEnterApp }: { onEnterApp: () => void }) {
  const [stats, setStats] = useState<StatsOverviewResponse | null>(null);

  useEffect(() => {
    getStatsOverview()
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  return (
    <div className="font-sans text-slate-900">
      <Hero onEnterApp={onEnterApp} stats={stats} />
      <HowItWorks />
      <Features />
      <FinalCta onEnterApp={onEnterApp} />
      <Footer />
    </div>
  );
}

function Hero({
  onEnterApp,
  stats,
}: {
  onEnterApp: () => void;
  stats: StatsOverviewResponse | null;
}) {
  return (
    <header className="relative overflow-hidden bg-slate-950 text-white">
      <AuroraBackground />
      <div className="absolute inset-0 bg-grid-pattern" aria-hidden />

      <nav className="relative z-10 mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-400 to-fuchsia-400 shadow-glow">
            <Sparkles className="h-4 w-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="font-display text-lg font-semibold tracking-tight">jobmarket</span>
        </div>
        <button
          onClick={onEnterApp}
          className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-medium text-white/90 backdrop-blur transition-colors hover:bg-white/10"
        >
          Launch app
        </button>
      </nav>

      <div className="relative z-10 mx-auto grid max-w-6xl gap-12 px-6 pb-24 pt-10 lg:grid-cols-2 lg:items-center lg:pb-32 lg:pt-16">
        <div className="animate-fade-in-up">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-indigo-400/30 bg-indigo-500/10 px-3 py-1 text-xs font-medium text-indigo-200">
            <Sparkles className="h-3 w-3" />
            Real market data, not a generic rewrite tool
          </span>
          <h1 className="font-display mt-5 text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
            AI-powered job market
            <span className="text-gradient block">intelligence</span>
          </h1>
          <p className="mt-5 max-w-lg text-lg leading-relaxed text-slate-300">
            Upload your CV, get matched against a real corpus of job postings, and see
            exactly which skills stand between you and your next role — with every
            claim grounded in your actual profile, never invented.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <button
              onClick={onEnterApp}
              className="group inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-6 py-3 text-sm font-semibold text-white shadow-glow transition-transform hover:scale-[1.03] active:scale-[0.98]"
            >
              Get started
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </button>
            <span className="text-sm text-slate-400">Free demo — no signup required</span>
          </div>

          {stats && (
            <dl className="mt-12 grid max-w-md grid-cols-3 gap-6 border-t border-white/10 pt-6">
              <StatTeaser value={stats.jobs.toLocaleString()} label="jobs indexed" />
              <StatTeaser value={stats.companies.toLocaleString()} label="companies" />
              <StatTeaser
                value={`${stats.embedding_coverage_pct.toFixed(0)}%`}
                label="semantically embedded"
              />
            </dl>
          )}
        </div>

        <div className="relative hidden animate-fade-in lg:block" style={{ animationDelay: "0.2s" }}>
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-indigo-500/10 to-fuchsia-500/10 blur-2xl" />
          <div className="relative rounded-3xl border border-white/10 bg-white/[0.03] p-8 backdrop-blur">
            <MatchGraphIllustration className="w-full" />
            <p className="mt-2 text-center text-xs text-slate-400">
              Lexical + semantic matching across your whole profile
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}

function StatTeaser({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="font-display text-2xl font-semibold text-white">{value}</div>
      <div className="text-xs text-slate-400">{label}</div>
    </div>
  );
}

const STEPS = [
  {
    icon: Upload,
    title: "Upload your CV",
    description:
      "Drop a PDF, DOCX, or TXT. Skills, experience, and education are extracted and grounded — nothing is guessed, and low-confidence results say so.",
  },
  {
    icon: Target,
    title: "Get matched",
    description:
      "Ranked against a real 36k-job corpus using lexical and semantic search, with a full breakdown of why each job scored the way it did.",
  },
  {
    icon: BarChart3,
    title: "See your gaps",
    description:
      "Missing skills ranked by real market demand, plus a data-driven CV rewrite that never claims a skill you don't have.",
  },
];

function HowItWorks() {
  return (
    <section className="bg-slate-50 py-24">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-display text-3xl font-semibold tracking-tight text-slate-900">
            How it works
          </h2>
          <p className="mt-3 text-slate-500">
            Three steps, all backed by data you can trace back to a real source.
          </p>
        </div>
        <div className="mt-14 grid gap-8 sm:grid-cols-3">
          {STEPS.map((step, index) => (
            <div
              key={step.title}
              className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-6 shadow-card transition-shadow hover:shadow-soft"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-glow">
                <step.icon className="h-5 w-5" strokeWidth={2} />
              </div>
              <h3 className="mt-4 font-display text-lg font-semibold text-slate-900">
                {step.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-500">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const FEATURES = [
  {
    icon: Database,
    title: "Real market data",
    description: "Every score, gap, and recommendation traces back to a real corpus of job postings.",
  },
  {
    icon: FileSearch,
    title: "Honest extraction",
    description:
      "Degraded or low-confidence extractions are flagged clearly, never hidden behind a polished result.",
  },
  {
    icon: Target,
    title: "Semantic + lexical matching",
    description: "Hybrid search surfaces jobs keyword search alone would miss.",
  },
  {
    icon: BarChart3,
    title: "Skill gap analysis",
    description: "See exactly which skills are missing, ranked by how often the market actually asks for them.",
  },
];

function Features() {
  return (
    <section className="border-t border-slate-100 bg-white py-24">
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid gap-x-8 gap-y-10 sm:grid-cols-2">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="flex gap-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
                <feature.icon className="h-5 w-5" strokeWidth={2} />
              </div>
              <div>
                <h3 className="font-display font-semibold text-slate-900">{feature.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-500">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function FinalCta({ onEnterApp }: { onEnterApp: () => void }) {
  return (
    <section className="relative overflow-hidden bg-slate-950 py-20 text-white">
      <AuroraBackground />
      <div className="relative z-10 mx-auto max-w-2xl px-6 text-center">
        <h2 className="font-display text-3xl font-semibold tracking-tight">
          See where you really stand
        </h2>
        <p className="mt-3 text-slate-300">
          Upload your CV and get a grounded, data-driven read on your fit — in under a minute.
        </p>
        <button
          onClick={onEnterApp}
          className="group mt-8 inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-6 py-3 text-sm font-semibold text-white shadow-glow transition-transform hover:scale-[1.03] active:scale-[0.98]"
        >
          Launch the app
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </button>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="bg-slate-950 py-8 text-center text-xs text-slate-500">
      <p>jobmarket — a data-driven CV matching platform. Built as an internship project.</p>
    </footer>
  );
}
