import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

/** Shared "hero-lite" header used across every inner app page for visual consistency
 * with the landing page's identity, without repeating a full dark hero section. */
export function PageHeader({
  icon: Icon,
  title,
  subtitle,
  action,
}: {
  icon: LucideIcon;
  title: string;
  subtitle: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex animate-fade-in-up flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-glow">
          <Icon className="h-6 w-6 text-white" strokeWidth={2} />
        </div>
        <div>
          <h2 className="font-display text-2xl font-semibold tracking-tight text-slate-900">
            {title}
          </h2>
          <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>
        </div>
      </div>
      {action}
    </div>
  );
}
