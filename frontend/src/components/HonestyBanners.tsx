/**
 * Surfaces the pipeline's honesty flags — extraction_degraded, out_of_scope,
 * low_confidence — as unmissable banners. This is deliberately the most visually
 * loud component in the app: these flags exist specifically so a bad result never
 * looks like a normal one, and that only works if the UI doesn't bury them.
 */

interface HonestyBannersProps {
  outOfScope?: boolean;
  outOfScopeMessage?: string | null;
  extractionDegraded?: boolean;
  extractionDegradedReason?: string | null;
  lowConfidence?: boolean;
  roughEstimate?: string | null;
  warnings?: string[];
  message?: string | null;
}

export function HonestyBanners({
  outOfScope,
  outOfScopeMessage,
  extractionDegraded,
  extractionDegradedReason,
  lowConfidence,
  roughEstimate,
  warnings = [],
  message,
}: HonestyBannersProps) {
  const hasAnyFlag = outOfScope || extractionDegraded || lowConfidence || Boolean(roughEstimate);
  if (!hasAnyFlag && warnings.length === 0) return null;

  return (
    <div className="space-y-2 mb-4" data-testid="honesty-banners">
      {outOfScope && (
        <Banner tone="red" badge="OUT OF SCOPE">
          {outOfScopeMessage ?? message ?? "This profile appears to fall outside the IT/tech scope this platform covers."}
        </Banner>
      )}
      {extractionDegraded && (
        <Banner tone="amber" badge="EXTRACTION DEGRADED">
          The richer LLM-based extraction failed, so a simpler deterministic parser
          was used instead — experience details may be missing or incomplete.
          {extractionDegradedReason && (
            <span className="block mt-1 text-xs opacity-80">
              Reason: {extractionDegradedReason}
            </span>
          )}
        </Banner>
      )}
      {lowConfidence && !outOfScope && (
        <Banner tone="amber" badge="LOW CONFIDENCE">
          CV extraction confidence is low — recommendations below are cautious
          estimates and worth a manual review.
        </Banner>
      )}
      {roughEstimate && (
        <Banner tone="blue" badge="ROUGH ESTIMATE">
          {roughEstimate}
        </Banner>
      )}
      {warnings.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {warnings.map((warning) => (
            <span
              key={warning}
              className="inline-block rounded bg-slate-200 px-2 py-0.5 text-xs font-mono text-slate-700"
            >
              {warning}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Banner({
  tone,
  badge,
  children,
}: {
  tone: "red" | "amber" | "blue";
  badge: string;
  children: React.ReactNode;
}) {
  const toneClasses =
    tone === "red"
      ? "bg-red-50 border-red-400 text-red-900"
      : tone === "amber"
        ? "bg-amber-50 border-amber-400 text-amber-900"
        : "bg-sky-50 border-sky-400 text-sky-900";
  const badgeClasses = tone === "red" ? "bg-red-600" : tone === "amber" ? "bg-amber-600" : "bg-sky-600";

  return (
    <div className={`flex items-start gap-3 rounded-lg border-2 px-4 py-3 ${toneClasses}`}>
      <span
        className={`shrink-0 rounded px-2 py-0.5 text-xs font-bold uppercase tracking-wide text-white ${badgeClasses}`}
      >
        {badge}
      </span>
      <p className="text-sm leading-snug">{children}</p>
    </div>
  );
}
