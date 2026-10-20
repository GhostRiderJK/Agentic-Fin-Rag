"use client";

import type { Citation } from "@/lib/api";

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations || citations.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic">No citations attached.</div>
    );
  }
  return (
    <ol className="space-y-1.5">
      {citations.map((c) => (
        <li key={c.chunk_id} className="text-xs border rounded-md p-2 bg-slate-50">
          <div className="flex items-center gap-2">
            <span className="font-mono font-bold text-brand-600">[{c.index}]</span>
            <span className="font-medium">
              {c.company || "Unknown"}
              {c.year ? ` · ${c.year}` : ""}
              {c.quarter ? ` ${c.quarter}` : ""}
            </span>
            <span className="ml-auto text-[10px] text-slate-500">
              {c.doc_type} · {c.page_number_type === "printed" ? `printed p.${c.printed_page ?? c.page}` : `PDF p.${c.physical_page ?? c.page}`}
              {c.page_number_type === "printed" && c.physical_page != null
                ? ` · PDF p.${c.physical_page}`
                : ""}
            </span>
          </div>
          {c.source_filename && (
            <div className="text-slate-500 mt-1 truncate" title={c.source_filename}>
              {c.source_filename}
            </div>
          )}
          <div className="text-slate-600 mt-1 line-clamp-3">{c.snippet}</div>
        </li>
      ))}
    </ol>
  );
}

export function VerificationBadge({ outcome }: { outcome: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    supported: { label: "Verified", cls: "bg-emerald-50 text-emerald-700" },
    partial: { label: "Partial", cls: "bg-amber-50 text-amber-700" },
    rejected: { label: "Rejected", cls: "bg-rose-50 text-rose-700" },
    not_applicable: { label: "N/A", cls: "bg-slate-100 text-slate-600" },
    no_context: { label: "No context", cls: "bg-slate-100 text-slate-600" },
    stub_not_evaluated: { label: "Test stub", cls: "bg-slate-100 text-slate-600" },
  };
  const v = map[outcome] || { label: outcome, cls: "bg-slate-100 text-slate-600" };
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${v.cls}`}>
      {v.label}
    </span>
  );
}
