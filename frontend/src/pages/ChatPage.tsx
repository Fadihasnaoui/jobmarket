import { type FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  Database,
  ExternalLink,
  Info,
  Loader2,
  MessageCircle,
  Send,
  Sparkles,
  UserRound,
} from "lucide-react";
import { ApiError, sendChatMessage } from "../api/client";
import type { ChatHistoryMessage, ChatJobReference, ChatResponse, ChatRoute } from "../api/types";
import { PageHeader } from "../components/PageHeader";
import { ErrorState } from "../components/StatusStates";

type ChatMessage =
  | { id: number; role: "user"; content: string }
  | {
      id: number;
      role: "assistant";
      content: string;
      route: ChatRoute;
      sources: ChatResponse["sources"];
    };

const EXAMPLE_QUESTIONS = [
  "How many jobs are in France?",
  "Which skills are most in demand?",
  "How does job matching work?",
];

const GREETING_RE = /^(?:hi|hello|hey)(?:\s+there)?[!,. ]*$/i;
const GREETING_ANSWER = "Hello! How can I help? I am the jobmarket AI assistant. Ask me about jobs, skills, salaries, or how this platform works.";

const MATCHES_PAGE_RE = /\b(?:match(?:es|ing)?|mathces)\b/i;
const PLATFORM_HELP_RE = /\b(?:page|website|app|explain|how|work|works)\b/i;
const MATCHES_PAGE_ANSWER = "The Matches page compares your CV with real job postings. It uses your skills, seniority, domains, and job preferences to rank roles. Each result shows a score breakdown, matched skills, missing important skills, and the reason it was recommended.";

function getStaticPlatformAnswer(question: string): string | null {
  if (GREETING_RE.test(question)) return GREETING_ANSWER;
  if (MATCHES_PAGE_RE.test(question) && PLATFORM_HELP_RE.test(question)) {
    return MATCHES_PAGE_ANSWER;
  }
  return null;
}

function normalizeForJobMatch(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function findMatchedJob(question: string, jobs: ChatJobReference[]): ChatJobReference | null {
  const normalizedQuestion = normalizeForJobMatch(question);
  const ranked = jobs
    .map((job) => {
      const title = normalizeForJobMatch(job.title);
      if (normalizedQuestion.includes(title)) return { job, score: 99 };
      const score = title
        .split(/[^a-z0-9]+/)
        .filter((word) => word.length >= 4)
        .filter((word) => normalizedQuestion.includes(word)).length;
      return { job, score };
    })
    .sort((left, right) => right.score - left.score);
  return ranked[0] && ranked[0].score >= 2 ? ranked[0].job : null;
}

const ROUTE_STYLES: Record<ChatRoute, { label: string; className: string }> = {
  sql: {
    label: "SQL - exact data",
    className: "border-emerald-200 bg-emerald-50 text-emerald-700",
  },
  rag: {
    label: "RAG - real job postings",
    className: "border-violet-200 bg-violet-50 text-violet-700",
  },
  platform: {
    label: "Platform - how it works",
    className: "border-sky-200 bg-sky-50 text-sky-700",
  },
  off_topic: {
    label: "Off-topic",
    className: "border-slate-200 bg-slate-100 text-slate-600",
  },
};

export function ChatPage({
  jobTarget,
  matchJobs,
  onJobTargetHandled,
}: {
  jobTarget: ChatJobReference | null;
  matchJobs: ChatJobReference[];
  onJobTargetHandled: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);
  const nextId = useRef(0);
  const historyEnd = useRef<HTMLDivElement | null>(null);
  const handledJobTarget = useRef<number | null>(null);

  useEffect(() => {
    historyEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending, error]);

  useEffect(() => {
    if (!jobTarget || handledJobTarget.current === jobTarget.jobId) return;
    handledJobTarget.current = jobTarget.jobId;
    onJobTargetHandled();
    void sendQuestion(
      "Tell me everything about the " + jobTarget.title + " job, including the description, salary, location, and contract.",
      true,
      jobTarget.jobId,
    );
  }, [jobTarget]);

  async function sendQuestion(rawQuestion: string, appendUserMessage = true, selectedJobId?: number) {
    const question = rawQuestion.trim();
    if (!question || isSending) return;

    const historySource = appendUserMessage ? messages : messages.slice(0, -1);
    const history: ChatHistoryMessage[] = historySource.slice(-6).map((message) => ({
      role: message.role,
      content: message.content,
    }));

    setError(null);
    setLastQuestion(question);
    if (appendUserMessage) {
      setMessages((current) => [
        ...current,
        { id: nextId.current++, role: "user", content: question },
      ]);
      setInput("");
    }

    const staticAnswer = getStaticPlatformAnswer(question);
    if (staticAnswer) {
      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: staticAnswer,
          route: "platform",
          sources: [],
        },
      ]);
      return;
    }

    const matchedJob = selectedJobId ? null : findMatchedJob(question, matchJobs);
    const jobId = selectedJobId ?? matchedJob?.jobId;
    setIsSending(true);

    try {
      const response = await sendChatMessage({ question, history, ...(jobId ? { job_id: jobId } : {}) });
      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: response.answer,
          route: response.route,
          sources: response.sources,
        },
      ]);
    } catch (requestError) {
      setError(requestError instanceof ApiError ? requestError.message : "Unable to send your message.");
    } finally {
      setIsSending(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendQuestion(input);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        icon={MessageCircle}
        title="jobmarket assistant"
        subtitle="Transparent answers grounded in corpus data or the way this app works."
      />

      <section className="animate-fade-in-up flex h-[min(66vh,640px)] min-h-[520px] flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white/90 shadow-card backdrop-blur">
        <div className="border-b border-slate-100 bg-gradient-to-r from-indigo-50/80 via-white to-fuchsia-50/70 px-5 py-3">
          <p className="flex items-center gap-2 text-xs font-medium text-slate-600">
            <Sparkles className="h-3.5 w-3.5 text-indigo-500" />
            Every answer shows its source: exact SQL data, real job postings, or how the platform works.
          </p>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-4 py-5 sm:px-6">
          {messages.length === 0 && (
            <EmptyChat onExample={(question) => void sendQuestion(question)} />
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <UserBubble key={message.id} content={message.content} />
            ) : (
              <AssistantBubble key={message.id} message={message} />
            ),
          )}

          {isSending && <TypingIndicator />}

          {error && (
            <ErrorState
              message={error}
              onRetry={lastQuestion ? () => void sendQuestion(lastQuestion, false) : undefined}
            />
          )}
          <div ref={historyEnd} />
        </div>

        <form onSubmit={handleSubmit} className="border-t border-slate-100 bg-white p-4 sm:p-5">
          <label htmlFor="chat-question" className="sr-only">
            Your question
          </label>
          <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-1.5 shadow-inner transition focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-100">
            <input
              id="chat-question"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about the job market..."
              disabled={isSending}
              className="min-w-0 flex-1 bg-transparent px-3 py-2 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!input.trim() || isSending}
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-glow transition hover:scale-[1.03] disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:scale-100"
              aria-label="Send message"
            >
              {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          </div>
          <p className="mt-2 px-2 text-xs text-slate-400">Press Enter to send. Conversation history stays only in this session.</p>
        </form>
      </section>
    </div>
  );
}

function EmptyChat({ onExample }: { onExample: (question: string) => void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center px-4 py-10 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-glow">
        <Bot className="h-7 w-7" strokeWidth={2} />
      </div>
      <h3 className="font-display mt-4 text-xl font-semibold text-slate-900">Hello! How can I help?</h3>
      <p className="mt-2 max-w-lg text-sm leading-relaxed text-slate-500">
        I am the jobmarket AI assistant. Ask me about job counts, in-demand skills, salaries, or how this platform works.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {EXAMPLE_QUESTIONS.map((question) => (
          <button
            key={question}
            onClick={() => onExample(question)}
            className="rounded-full border border-indigo-100 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 transition hover:border-indigo-200 hover:bg-indigo-100"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end gap-3">
      <div className="max-w-[82%] whitespace-pre-wrap break-words rounded-2xl rounded-tr-sm bg-gradient-to-br from-indigo-500 to-violet-500 px-4 py-3 text-sm leading-relaxed text-white shadow-sm">
        {content}
      </div>
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-indigo-100 text-indigo-600">
        <UserRound className="h-4 w-4" />
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: Extract<ChatMessage, { role: "assistant" }> }) {
  const route = ROUTE_STYLES[message.route];
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-fuchsia-100 text-fuchsia-600">
        <Bot className="h-4 w-4" />
      </div>
      <div className="max-w-[82%] rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <span className={["inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold", route.className].join(" ")}>
          {message.route === "sql" ? <Database className="h-3 w-3" /> : <Info className="h-3 w-3" />}
          {route.label}
        </span>
        <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700">{message.content}</p>
        {message.sources.length > 0 && (
          <div className="mt-3 border-t border-slate-100 pt-3">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Cited jobs</p>
            <ul className="space-y-1.5">
              {message.sources.map((source) => (
                <li key={[source.label, source.url].join("-")} className="text-xs">
                  {source.url ? (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-indigo-600 hover:text-indigo-800 hover:underline"
                    >
                      {source.label}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  ) : (
                    <span className="text-slate-600">{source.label}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-fuchsia-100 text-fuchsia-600">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm" aria-label="The assistant is responding">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-400 [animation-delay:150ms]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-fuchsia-400 [animation-delay:300ms]" />
      </div>
    </div>
  );
}
