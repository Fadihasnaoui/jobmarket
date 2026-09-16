import { AlertTriangle, Loader2 } from "lucide-react";
import { EmptyDocumentIllustration } from "./illustrations";

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex animate-fade-in items-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-7 shadow-card">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500">
        <Loader2 className="h-4 w-4 animate-spin text-white" strokeWidth={2.5} />
      </span>
      <span className="text-sm font-medium text-slate-600">{label}</span>
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex animate-fade-in items-start justify-between gap-4 rounded-2xl border border-red-200 bg-red-50 px-5 py-5 text-red-900 shadow-card">
      <div className="flex gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-500" strokeWidth={2} />
        <div>
          <p className="text-sm font-semibold">Something went wrong</p>
          <p className="mt-1 text-sm text-red-800/80">{message}</p>
        </div>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="shrink-0 rounded-full bg-red-600 px-4 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-red-700"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex animate-fade-in flex-col items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-white/70 px-6 py-14 text-center shadow-card">
      <EmptyDocumentIllustration className="h-20 w-20 opacity-90" />
      <p className="max-w-sm text-sm text-slate-500">{children}</p>
    </div>
  );
}
