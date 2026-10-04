"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatResponse } from "@/lib/api";
import { CitationList, VerificationBadge } from "./CitationList";

export function Answer({ resp }: { resp: ChatResponse }) {
  return (
    <div className="space-y-3">
      <div className="prose-chat text-slate-800">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {resp.answer || "_No answer produced._"}
        </ReactMarkdown>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <VerificationBadge outcome={String(resp.verification?.outcome || "")} />
        <span className="text-slate-500">
          {resp.latency_ms} ms
        </span>
        <span className="text-slate-500">
          {String(resp.retrieval?.n_chunks ?? 0)} chunks
        </span>
        {resp.tool_results && resp.tool_results.length > 0 && (
          <span className="text-slate-500">
            {resp.tool_results.length} tool call(s)
          </span>
        )}
        {(resp.plan as { is_metric_lookup?: boolean })?.is_metric_lookup && (
          <span className="text-brand-600">metric route</span>
        )}
        {(resp.plan as { is_comparison?: boolean })?.is_comparison && (
          <span className="text-brand-600">comparison</span>
        )}
      </div>

      <details className="border rounded-md">
        <summary className="cursor-pointer text-xs px-3 py-2 bg-slate-50 font-medium">
          Sources ({resp.citations.length})
        </summary>
        <div className="p-3">
          <CitationList citations={resp.citations} />
        </div>
      </details>

      {(resp.tool_results?.length ?? 0) > 0 && (
        <details className="border rounded-md">
          <summary className="cursor-pointer text-xs px-3 py-2 bg-slate-50 font-medium">
            Tool calls ({resp.tool_results.length})
          </summary>
          <div className="p-3 space-y-2">
            {resp.tool_results.map((t, i) => (
              <pre key={i} className="text-[11px] bg-slate-900 text-slate-100 p-2 rounded overflow-x-auto">
                {JSON.stringify(t, null, 2)}
              </pre>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
