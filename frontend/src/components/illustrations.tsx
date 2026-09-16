/**
 * Hand-authored, generated SVG illustrations — no external image assets, no
 * copyrighted material. Purely decorative; safe to omit for a11y (aria-hidden).
 */

export function AuroraBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      <div className="absolute -top-32 left-1/4 h-96 w-96 animate-float-slow rounded-full bg-indigo-500/30 blur-3xl" />
      <div
        className="absolute top-1/3 -right-20 h-80 w-80 animate-float rounded-full bg-fuchsia-500/20 blur-3xl"
        style={{ animationDelay: "1.5s" }}
      />
      <div
        className="absolute bottom-0 left-1/3 h-72 w-72 animate-float-slow rounded-full bg-violet-500/20 blur-3xl"
        style={{ animationDelay: "3s" }}
      />
    </div>
  );
}

/**
 * Light, low-opacity variant of the hero's aurora + grid, for the app shell behind
 * Upload/Matches/Skill Gap/Dashboard. Pastel tints instead of saturated blobs, and a
 * dark-on-light grid instead of light-on-dark — the goal is an ambient backdrop that
 * ties these pages to the landing page's identity without competing with card text.
 * Fixed positioning: one persistent backdrop behind the whole app shell, not per-page.
 *
 * Pure-SVG/CSS variant — no external image. Kept as an easy fallback alongside
 * `AppBackgroundPhoto` below in case the photo variant hurts readability.
 */
export function AppBackgroundSvg() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-slate-50" aria-hidden>
      <div className="absolute inset-0 bg-grid-pattern-light" />
      <div className="absolute -top-40 left-[15%] h-[28rem] w-[28rem] rounded-full bg-indigo-200/50 blur-3xl" />
      <div className="absolute top-1/4 -right-32 h-96 w-96 rounded-full bg-fuchsia-200/40 blur-3xl" />
      <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-violet-200/40 blur-3xl" />
      <MatchGraphIllustration className="absolute -right-6 top-20 h-72 w-72 opacity-[0.06]" />
    </div>
  );
}

/**
 * Real-photo variant: a free-licensed Pexels photo (see public/images/CREDITS.md for
 * source/photographer/license) behind a light overlay that knocks the visible image
 * down to roughly 15-20% — the overlay's own color (not the image's CSS opacity)
 * does that, which keeps the tint controllable and avoids the photo reading as flat
 * gray. The brand-color blobs, grid, and node-graph accent are layered on top of the
 * photo, same as the SVG-only variant, so the two look like the same design system.
 */
export function AppBackgroundPhoto() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-slate-50" aria-hidden>
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: "url(/images/network-background.jpg)" }}
      />
      <div className="absolute inset-0 bg-slate-50/85" />
      <div className="absolute inset-0 bg-grid-pattern-light" />
      <div className="absolute -top-40 left-[15%] h-[28rem] w-[28rem] rounded-full bg-indigo-200/40 blur-3xl" />
      <div className="absolute top-1/4 -right-32 h-96 w-96 rounded-full bg-fuchsia-200/30 blur-3xl" />
      <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-violet-200/30 blur-3xl" />
      <MatchGraphIllustration className="absolute -right-6 top-20 h-72 w-72 opacity-[0.06]" />
    </div>
  );
}

/** Abstract nodes-and-edges graphic evoking matching/graph search — the hero's visual anchor. */
export function MatchGraphIllustration({ className = "" }: { className?: string }) {
  const nodes = [
    { x: 60, y: 60, r: 7, fill: "url(#node-grad-a)" },
    { x: 200, y: 40, r: 5, fill: "#a5b4fc" },
    { x: 280, y: 110, r: 9, fill: "url(#node-grad-b)" },
    { x: 150, y: 150, r: 6, fill: "#c4b5fd" },
    { x: 40, y: 190, r: 5, fill: "#818cf8" },
    { x: 240, y: 210, r: 6, fill: "#e879f9" },
    { x: 320, y: 190, r: 4, fill: "#a5b4fc" },
  ];
  const edges: [number, number][] = [
    [0, 1],
    [1, 2],
    [0, 3],
    [3, 2],
    [3, 4],
    [3, 5],
    [2, 6],
    [5, 6],
  ];
  return (
    <svg viewBox="0 0 360 260" className={className} aria-hidden>
      <defs>
        <linearGradient id="node-grad-a" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#6366f1" />
        </linearGradient>
        <linearGradient id="node-grad-b" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#e879f9" />
          <stop offset="100%" stopColor="#a855f7" />
        </linearGradient>
      </defs>
      {edges.map(([a, b], i) => (
        <line
          key={i}
          x1={nodes[a].x}
          y1={nodes[a].y}
          x2={nodes[b].x}
          y2={nodes[b].y}
          stroke="url(#node-grad-a)"
          strokeOpacity={0.35}
          strokeWidth={1.5}
        />
      ))}
      {nodes.map((n, i) => (
        <circle key={i} cx={n.x} cy={n.y} r={n.r} fill={n.fill} className="drop-shadow-sm" />
      ))}
    </svg>
  );
}

export function EmptyDocumentIllustration({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 120 120" className={className} aria-hidden>
      <rect x="28" y="14" width="64" height="92" rx="8" fill="#eef2ff" stroke="#c7d2fe" strokeWidth="2" />
      <line x1="40" y1="36" x2="80" y2="36" stroke="#a5b4fc" strokeWidth="3" strokeLinecap="round" />
      <line x1="40" y1="50" x2="80" y2="50" stroke="#c7d2fe" strokeWidth="3" strokeLinecap="round" />
      <line x1="40" y1="64" x2="64" y2="64" stroke="#c7d2fe" strokeWidth="3" strokeLinecap="round" />
      <circle cx="88" cy="88" r="20" fill="#f5f3ff" stroke="#ddd6fe" strokeWidth="2" />
      <path d="M80 88h16M88 80v16" stroke="#a855f7" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
