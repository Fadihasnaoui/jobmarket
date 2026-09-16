import { useState } from "react";
import { BarChart3, ClipboardCheck, FileUp, MessageCircle, Sparkles, Target } from "lucide-react";
import { CvSessionProvider } from "./context/CvSessionContext";
import { AppBackgroundPhoto } from "./components/illustrations";
import { LandingPage } from "./pages/LandingPage";
import { UploadPage } from "./pages/UploadPage";
import { MatchesPage } from "./pages/MatchesPage";
import { SkillGapPage } from "./pages/SkillGapPage";
import { QualityReportPage } from "./pages/QualityReportPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ChatPage } from "./pages/ChatPage";
import type { ChatJobReference } from "./api/types";

type View = "landing" | "app";
type Tab = "upload" | "matches" | "skill-gap" | "quality" | "chat" | "dashboard";

const TABS: { id: Tab; label: string; icon: typeof FileUp }[] = [
  { id: "upload", label: "Upload", icon: FileUp },
  { id: "matches", label: "Matches", icon: Target },
  { id: "skill-gap", label: "Skill Gap", icon: Sparkles },
  { id: "quality", label: "Quality Report", icon: ClipboardCheck },
  { id: "chat", label: "Chat", icon: MessageCircle },
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
];

function App() {
  const [view, setView] = useState<View>("landing");
  const [tab, setTab] = useState<Tab>("upload");
  const [matchJobs, setMatchJobs] = useState<ChatJobReference[]>([]);
  const [chatJobTarget, setChatJobTarget] = useState<ChatJobReference | null>(null);

  if (view === "landing") {
    return <LandingPage onEnterApp={() => setView("app")} />;
  }

  return (
    <CvSessionProvider>
      <div className="relative min-h-screen">
        <AppBackgroundPhoto />
        <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/80 backdrop-blur">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <button
              onClick={() => setView("landing")}
              className="flex items-center gap-2 transition-opacity hover:opacity-80"
            >
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-fuchsia-500">
                <Sparkles className="h-3.5 w-3.5 text-white" strokeWidth={2.5} />
              </div>
              <span className="font-display text-base font-semibold tracking-tight text-slate-900">
                jobmarket
              </span>
            </button>
            <nav className="flex gap-1 rounded-full bg-slate-100 p-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-medium transition-all ${
                    tab === t.id
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  <t.icon className="h-3.5 w-3.5" strokeWidth={2.25} />
                  {t.label}
                </button>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">
          {tab === "upload" && <UploadPage onDone={() => setTab("matches")} />}
          {tab === "matches" && (
  <MatchesPage
    onMatchesLoaded={setMatchJobs}
    onAskAboutJob={(job) => {
      setChatJobTarget(job);
      setTab("chat");
    }}
  />
)}
          {tab === "skill-gap" && <SkillGapPage />}
          {tab === "quality" && <QualityReportPage />}
          {tab === "chat" && (
  <ChatPage
    jobTarget={chatJobTarget}
    matchJobs={matchJobs}
    onJobTargetHandled={() => setChatJobTarget(null)}
  />
)}
          {tab === "dashboard" && <DashboardPage />}
        </main>
      </div>
    </CvSessionProvider>
  );
}

export default App;
