import { useRef, useState } from "react";
import {
  Briefcase,
  CheckCircle2,
  GraduationCap,
  HelpCircle,
  Layers,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { ApiError, uploadCv } from "../api/client";
import { HonestyBanners } from "../components/HonestyBanners";
import { PageHeader } from "../components/PageHeader";
import { ErrorState, LoadingState } from "../components/StatusStates";
import { useCvSession } from "../context/CvSessionContext";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt"];

export function UploadPage({ onDone }: { onDone: () => void }) {
  const { upload, setUpload } = useCvSession();
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    const extension = "." + (file.name.split(".").pop() ?? "").toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setError(`Unsupported file type "${extension}". Use .pdf, .docx, or .txt.`);
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const response = await uploadCv(file);
      setUpload(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed unexpectedly.");
    } finally {
      setIsLoading(false);
    }
  }

  function onDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        icon={UploadCloud}
        title="Upload a CV"
        subtitle="Drop a PDF, DOCX, or TXT file — it's parsed, extracted, and embedded once. Nothing is stored beyond a short-lived session."
      />

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`group relative cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed px-6 py-16 text-center transition-all ${
          isDragging
            ? "border-indigo-400 bg-indigo-50/60 shadow-glow"
            : "border-slate-300 bg-white shadow-card hover:border-indigo-300 hover:bg-indigo-50/30"
        }`}
      >
        <div
          className={`mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-glow transition-transform ${
            isDragging ? "scale-110" : "group-hover:-translate-y-0.5"
          }`}
        >
          <UploadCloud className="h-7 w-7 text-white" strokeWidth={2} />
        </div>
        <p className="mt-4 text-sm font-medium text-slate-700">
          Drag &amp; drop a CV here, or click to browse
        </p>
        <p className="mt-1 text-xs text-slate-400">.pdf, .docx, .txt</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
            e.target.value = "";
          }}
        />
      </div>

      {isLoading && <LoadingState label="Parsing, extracting, and embedding the CV…" />}
      {error && <ErrorState message={error} onRetry={() => setError(null)} />}

      {upload && !isLoading && (
        <ExtractedProfilePanel upload={upload} onViewMatches={onDone} />
      )}
    </div>
  );
}

function ExtractedProfilePanel({
  upload,
  onViewMatches,
}: {
  upload: ReturnType<typeof useCvSession>["upload"];
  onViewMatches: () => void;
}) {
  if (!upload) return null;
  const profile = upload.profile;

  if (!profile) {
    return (
      <ErrorState
        message={upload.message ?? "The document could not be parsed into a profile."}
      />
    );
  }

  return (
    <div className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-6 shadow-card">
      <HonestyBanners
        outOfScope={upload.out_of_scope}
        outOfScopeMessage={upload.message}
        extractionDegraded={upload.extraction_degraded}
        extractionDegradedReason={upload.extraction_degraded_reason}
        lowConfidence={profile.extraction_confidence_label === "low"}
        warnings={upload.warnings}
        message={upload.message}
      />

      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg font-semibold text-slate-900">Extracted profile</h3>
        <ConfidenceBadge
          label={profile.extraction_confidence_label}
          score={profile.extraction_confidence_score}
        />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <Field label="Career level" value={profile.career_level} />
        <Field label="Extraction method" value={upload.extraction_method} />
        <Field label="Filename" value={profile.filename} />
      </dl>

      <Section icon={Sparkles} title="Skills">
        <TagList items={profile.canonical_skills} empty="No skills extracted." tone="indigo" />
      </Section>

      <Section icon={Layers} title="Domains">
        <TagList items={profile.domains} empty="No domains inferred." tone="slate" />
      </Section>

      <Section icon={GraduationCap} title="Education">
        {profile.education ? (
          <p className="text-sm text-slate-700">
            {[
              profile.education.education_level,
              profile.education.education_field,
              profile.education.institution,
            ]
              .filter(Boolean)
              .join(" · ") || "Education detected, no further detail."}
            {profile.education.graduation_year && (
              <span className="text-slate-400"> ({profile.education.graduation_year})</span>
            )}
          </p>
        ) : (
          <p className="text-sm text-slate-400">No education detected.</p>
        )}
      </Section>

      <Section icon={Briefcase} title={`Experience (${profile.experiences.length})`}>
        {profile.experiences.length === 0 ? (
          <p className="text-sm text-slate-400">No experience entries detected.</p>
        ) : (
          <ul className="space-y-2">
            {profile.experiences.map((entry, index) => (
              <li
                key={index}
                className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-3.5 py-2.5 text-sm transition-colors hover:bg-slate-100/70"
              >
                <span className="min-w-0">
                  <span className="font-medium text-slate-800">{entry.title ?? "Untitled role"}</span>
                  {entry.employer && <span className="text-slate-500"> · {entry.employer}</span>}
                </span>
                <span className="ml-2 flex shrink-0 items-center gap-1.5">
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                      entry.duration_months === null
                        ? "bg-slate-100 text-slate-500"
                        : "bg-violet-100 text-violet-700"
                    }`}
                  >
                    {formatExperienceDuration(entry.duration_months)}
                  </span>
                  {entry.entry_type === "internship" && (
                    <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-semibold text-indigo-700">
                      internship
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <div className="mt-6">
        <button
          onClick={onViewMatches}
          className="group inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-5 py-2.5 text-sm font-semibold text-white shadow-glow transition-transform hover:scale-[1.02] active:scale-[0.98]"
        >
          View job matches
          <span className="transition-transform group-hover:translate-x-0.5">→</span>
        </button>
      </div>
    </div>
  );
}

function formatExperienceDuration(months: number | null): string {
  if (months === null) return "Duration unavailable";
  if (months < 1) return "Less than 1 month";

  const years = Math.floor(months / 12);
  const remainingMonths = months % 12;
  const parts = [
    years > 0 ? `${years} ${years === 1 ? "year" : "years"}` : null,
    remainingMonths > 0
      ? `${remainingMonths} ${remainingMonths === 1 ? "month" : "months"}`
      : null,
  ].filter(Boolean);

  return parts.join(" ");
}

function ConfidenceBadge({ label, score }: { label: string | null; score: number | null }) {
  const config =
    label === "high"
      ? { classes: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200", Icon: CheckCircle2 }
      : label === "low"
        ? { classes: "bg-amber-50 text-amber-700 ring-1 ring-amber-200", Icon: HelpCircle }
        : { classes: "bg-slate-100 text-slate-600 ring-1 ring-slate-200", Icon: HelpCircle };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${config.classes}`}
    >
      <config.Icon className="h-3.5 w-3.5" strokeWidth={2} />
      confidence: {label ?? "unknown"} ({score !== null ? `${Math.round(score * 100)}%` : "n/a"})
    </span>
  );
}

function Field({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-xl bg-slate-50 px-3 py-2">
      <dt className="text-xs uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="font-medium text-slate-800">{value ?? "unknown"}</dd>
    </div>
  );
}

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof Sparkles;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-5">
      <h4 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-slate-600">
        <Icon className="h-4 w-4 text-indigo-500" strokeWidth={2} />
        {title}
      </h4>
      {children}
    </div>
  );
}

function TagList({
  items,
  empty,
  tone,
}: {
  items: string[];
  empty: string;
  tone: "indigo" | "slate";
}) {
  if (items.length === 0) return <p className="text-sm text-slate-400">{empty}</p>;
  const toneClasses =
    tone === "indigo"
      ? "bg-indigo-50 text-indigo-700 ring-1 ring-indigo-100"
      : "bg-slate-100 text-slate-600 ring-1 ring-slate-200";
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <span
          key={item}
          className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${toneClasses}`}
        >
          {item}
        </span>
      ))}
    </div>
  );
}
