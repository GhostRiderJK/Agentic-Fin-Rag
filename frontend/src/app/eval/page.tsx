"use client";

import { useEffect, useRef, useState } from "react";
import { useApp } from "@/components/AppShell";
import { ConnectionGate } from "@/components/ConnectionGate";
import { WorkspacePicker } from "@/components/WorkspacePicker";
import { api, type EvalRun, type EvalSuite, type Job } from "@/lib/api";

const METRIC_LABELS: Record<string, string> = {
  faithfulness: "Faithfulness",
  answer_relevance: "Answer relevance",
  context_precision: "Context precision",
  context_recall: "Context recall",
  citation_precision: "Citation precision",
  numeric_accuracy: "Numeric accuracy",
};

// A suite run takes minutes, so it is polled rather than awaited: a single
// request held open that long is cut off by the Next /api proxy at 30s.
const POLL_MS = 2000;

export default function EvalPage() {
  const { apiKey, workspaceId } = useApp();
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [suites, setSuites] = useState<EvalSuite[]>([]);
  const [suiteId, setSuiteId] = useState("packaged");
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  async function refresh() {
    if (!apiKey) return;
    try {
      setRuns(await api.listEvalRuns(apiKey));
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  useEffect(() => {
    if (!apiKey) return;
    api.listEvalSuites(apiKey)
      .then((s) => {
        setSuites(s);
        setSuiteId((cur) => (s.some((x) => x.id === cur) ? cur : s[0]?.id ?? "packaged"));
      })
      .catch(() => setSuites([]));
    refresh();
  }, [apiKey]);

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  if (!apiKey) return <ConnectionGate />;

  function stopPolling() {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
    setBusy(false);
  }

  async function run() {
    if (!apiKey || !workspaceId || busy) return;
    setBusy(true); setErr(null); setJob(null);
    try {
      const accepted = await api.runEval(apiKey, workspaceId, suiteId);
      pollRef.current = setInterval(async () => {
        if (!apiKey) return;
        let j: Job;
        try {
          j = await api.getJob(apiKey, accepted.job_id);
        } catch (e) {
          setErr((e as Error).message);
          stopPolling();
          return;
        }
        setJob(j);
        if (j.status === "succeeded" || j.status === "failed") {
          stopPolling();
          if (j.status === "failed") setErr(j.error || "eval job failed");
          await refresh();
        }
      }, POLL_MS);
    } catch (e) {
      setErr((e as Error).message);
      stopPolling();
    }
  }

  const finished = job?.status === "succeeded" ? job : null;
  const liveMetrics = (finished?.result?.metrics ?? {}) as Record<string, number>;
  const hasLive = Object.keys(liveMetrics).length > 0;
  const latest = runs[0];
  const metrics = hasLive ? liveMetrics : latest?.metrics ?? {};
  const meanKeys = Object.keys(metrics).filter((k) => k.startsWith("mean_"));
  const shownSuite = hasLive
    ? String(finished?.result?.suite ?? suiteId)
    : latest?.suite;
  const shownItems = hasLive
    ? Number(finished?.result?.n_items ?? 0)
    : latest?.n_items;
  // A run whose requests all failed scores nothing: the backend leaves the
  // mean_* keys out rather than averaging the error text. Say so instead of
  // showing the "no runs yet" panel, which would read as an untouched app.
  const nErrored = Number(
    metrics.n_errored ?? finished?.result?.n_errored ?? 0);
  const hasRun = hasLive || Boolean(latest);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-12 gap-4">
      <aside className="col-span-3 space-y-3">
        <WorkspacePicker />
        <div className="bg-white border rounded-lg p-3 space-y-2">
          <div className="text-xs font-medium text-slate-500">Evaluation</div>
          <p className="text-[11px] text-slate-500">
            Runs a gold Q&amp;A suite through the full pipeline and records
            faithfulness, retrieval, citation, and numeric-accuracy metrics.
          </p>

          <label className="block text-[11px] text-slate-500">
            Suite
            <select
              value={suiteId}
              onChange={(e) => setSuiteId(e.target.value)}
              disabled={busy}
              className="mt-1 w-full border rounded px-2 py-1 text-xs bg-white disabled:opacity-50"
            >
              {suites.length === 0 && <option value="packaged">packaged</option>}
              {suites.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} — {s.n_items} items{s.packaged ? " (seeded)" : ""}
                </option>
              ))}
            </select>
          </label>

          <button onClick={run} disabled={busy || !workspaceId}
            className="w-full bg-brand-600 text-white text-sm py-1.5 rounded hover:bg-brand-700 disabled:opacity-50">
            {busy ? "Running…" : "Run eval suite"}
          </button>

          {busy && (
            <div className="text-[11px] text-slate-500">
              job {job?.status ?? "queued"} — this takes a few minutes; you can
              leave this page, the run continues.
            </div>
          )}
        </div>
      </aside>

      <section className="col-span-9 space-y-4">
        <h1 className="text-xl font-semibold">Evaluation dashboard</h1>
        {err && (
          <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded p-2">{err}</div>
        )}

        {meanKeys.length > 0 ? (
          <div className="bg-white border rounded-lg p-4">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-semibold">
                {hasLive ? "Latest run" : "Previous run"}
              </h2>
              <div className="text-xs text-slate-500">
                {shownSuite} · {shownItems} items
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
            {nErrored > 0 && (
              <div className="mt-3 text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                {nErrored} of {shownItems} items never answered, so they are
                excluded from the averages above. Check the model pins and the
                gateway credentials before reading these numbers as quality.
              </div>
            )}
          </div>
        ) : (
          <div className="bg-white border rounded-lg p-8 text-center text-slate-500 text-sm">
            {hasRun
              ? "The recorded run scored no items — every request failed before " +
                "answering. Nothing here is a quality number."
              : "No evaluation runs yet. Pick a suite and click “Run eval suite”."}
          </div>
        )}

        {runs.length > 0 && (
          <div className="bg-white border rounded-lg overflow-hidden">
            <div className="p-3 text-sm font-medium border-b">History</div>
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-xs text-slate-500">
                <tr>
                  <th className="text-left p-2">When</th>
                  <th className="text-left p-2">Suite</th>
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
                    <td className="p-2 text-xs">{r.suite}</td>
                    <td className="p-2 text-xs">
                      {r.n_items}
                      {r.n_items > 0 && Number(r.metrics.n_scored ?? r.n_items) < r.n_items && (
                        <span className="text-slate-400">
                          {` (${Number(r.metrics.n_scored ?? 0)} scored)`}
                        </span>
                      )}
                    </td>
                    <td className="p-2 text-xs">{fmtMean(r.metrics.mean_faithfulness)}</td>
                    <td className="p-2 text-xs">{fmtMean(r.metrics.mean_numeric_accuracy)}</td>
                    <td className="p-2 text-xs">{fmtMean(r.metrics.mean_citation_precision)}</td>
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

// A metric the run never produced must not render as 0.00: that reads as a
// failing score, not as no score, and the whole point of the accounting fix is
// that an unanswered request is not a measurement.
function fmtMean(v: number | undefined) {
  return v === undefined ? "—" : v.toFixed(2);
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
