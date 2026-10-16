"use client";

import { useEffect, useState } from "react";
import { useApp } from "@/components/AppShell";
import { ConnectionGate } from "@/components/ConnectionGate";
import { WorkspacePicker } from "@/components/WorkspacePicker";
import { api, type EvalRun } from "@/lib/api";

const METRIC_LABELS: Record<string, string> = {
  faithfulness: "Faithfulness",
  answer_relevance: "Answer relevance",
  context_precision: "Context precision",
  context_recall: "Context recall",
  citation_precision: "Citation precision",
  numeric_accuracy: "Numeric accuracy",
};

export default function EvalPage() {
  const { apiKey, workspaceId } = useApp();
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function refresh() {
    if (!apiKey) return;
    try {
      const r = await api.listEvalRuns(apiKey);
      setRuns(r);
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  useEffect(() => { refresh(); }, [apiKey]);

  if (!apiKey) return <ConnectionGate />;

  async function run() {
    if (!apiKey || !workspaceId) return;
    setBusy(true); setErr(null);
    try {
      await api.runEval(apiKey, workspaceId);
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const latest = runs[0];
  const metrics = latest?.metrics ?? {};
  const meanKeys = Object.keys(metrics).filter((k) => k.startsWith("mean_"));

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-12 gap-4">
      <aside className="col-span-3 space-y-3">
        <WorkspacePicker />
        <div className="bg-white border rounded-lg p-3 space-y-2">
          <div className="text-xs font-medium text-slate-500">Evaluation</div>
          <p className="text-[11px] text-slate-500">
            Runs the gold Q&amp;A suite through the full pipeline and records
            faithfulness, retrieval, citation, and numeric-accuracy metrics.
          </p>
          <button onClick={run} disabled={busy || !workspaceId}
            className="w-full bg-brand-600 text-white text-sm py-1.5 rounded hover:bg-brand-700 disabled:opacity-50">
            {busy ? "Running…" : "Run eval suite"}
          </button>
        </div>
      </aside>

      <section className="col-span-9 space-y-4">
        <h1 className="text-xl font-semibold">Evaluation dashboard</h1>
        {err && (
          <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded p-2">{err}</div>
        )}

        {latest ? (
          <>
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-baseline justify-between mb-3">
                <h2 className="font-semibold">Latest run</h2>
                <div className="text-xs text-slate-500">
                  {new Date(latest.created_at).toLocaleString()} · {latest.n_items} items
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {meanKeys.map((k) => {
                  const key = k.replace("mean_", "");
                  const mean = metrics[k];
                  const pass = metrics[`pass_${key}`] ?? 0;
                  return (
                    <div key={k} className="border rounded-lg p-3">
                      <div className="text-xs text-slate-500">
                        {METRIC_LABELS[key] || key}
                      </div>
                      <div className={`text-2xl font-semibold ${scoreColor(mean)}`}>
                        {mean.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-1">
                        pass-rate {(pass * 100).toFixed(0)}%
                      </div>
                      <Bar value={mean} />
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        ) : (
          <div className="bg-white border rounded-lg p-8 text-center text-slate-500 text-sm">
            No evaluation runs yet. Click “Run eval suite” to start.
          </div>
        )}

        {runs.length > 0 && (
          <div className="bg-white border rounded-lg overflow-hidden">
            <div className="p-3 text-sm font-medium border-b">History</div>
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-xs text-slate-500">
                <tr>
                  <th className="text-left p-2">When</th>
                  <th className="text-left p-2">Items</th>
                  <th className="text-left p-2">Faithfulness</th>
                  <th className="text-left p-2">Numeric acc.</th>
                  <th className="text-left p-2">Citation prec.</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} className="border-t">
                    <td className="p-2 text-xs">{new Date(r.created_at).toLocaleString()}</td>
                    <td className="p-2 text-xs">{r.n_items}</td>
                    <td className="p-2 text-xs">{(r.metrics.mean_faithfulness ?? 0).toFixed(2)}</td>
                    <td className="p-2 text-xs">{(r.metrics.mean_numeric_accuracy ?? 0).toFixed(2)}</td>
                    <td className="p-2 text-xs">{(r.metrics.mean_citation_precision ?? 0).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function scoreColor(v: number) {
  if (v >= 0.8) return "text-emerald-600";
  if (v >= 0.5) return "text-amber-600";
  return "text-rose-600";
}

function Bar({ value }: { value: number }) {
  return (
    <div className="h-1.5 bg-slate-100 rounded mt-2 overflow-hidden">
      <div className={`h-full ${value >= 0.8 ? "bg-emerald-500" : value >= 0.5 ? "bg-amber-500" : "bg-rose-500"}`}
        style={{ width: `${Math.min(100, value * 100)}%` }} />
    </div>
  );
}
