"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useApp } from "@/components/AppShell";
import { ConnectionGate } from "@/components/ConnectionGate";
import { WorkspacePicker } from "@/components/WorkspacePicker";
import { api, type Job } from "@/lib/api";

export default function ReportsPage() {
  const { apiKey, workspaceId } = useApp();
  const [company, setCompany] = useState("ACME");
  const [year, setYear] = useState("2025");
  const [reportType, setReportType] = useState<"summary" | "thesis">("summary");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [openJob, setOpenJob] = useState<Job | null>(null);

  async function refresh() {
    if (!apiKey || !workspaceId) return;
    try {
      const j = await api.listJobs(apiKey, workspaceId);
      setJobs(j.filter((x) => x.kind === "report"));
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [apiKey, workspaceId]);

  if (!apiKey) return <ConnectionGate />;

  async function generate() {
    if (!apiKey || !workspaceId) return;
    setBusy(true); setErr(null);
    try {
      await api.createReport(
        apiKey, workspaceId, company, reportType,
        year ? parseInt(year, 10) : undefined,
      );
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-12 gap-4">
      <aside className="col-span-3 space-y-3">
        <WorkspacePicker />
        <div className="bg-white border rounded-lg p-3 space-y-2">
          <div className="text-xs font-medium text-slate-500">Generate report</div>
          <input value={company} onChange={(e) => setCompany(e.target.value)}
            placeholder="Company" className="w-full border rounded-md px-2 py-1.5 text-sm" />
          <input value={year} onChange={(e) => setYear(e.target.value)}
            placeholder="Year" className="w-full border rounded-md px-2 py-1.5 text-sm" />
          <select value={reportType}
            onChange={(e) => setReportType(e.target.value as "summary" | "thesis")}
            className="w-full border rounded-md px-2 py-1.5 text-sm">
            <option value="summary">Earnings summary</option>
            <option value="thesis">Investment thesis</option>
          </select>
          <button onClick={generate} disabled={busy || !workspaceId}
            className="w-full bg-brand-600 text-white text-sm py-1.5 rounded hover:bg-brand-700 disabled:opacity-50">
            {busy ? "Submitting…" : "Generate"}
          </button>
        </div>
      </aside>

      <section className="col-span-9 space-y-4">
        <h1 className="text-xl font-semibold">Reports</h1>
        {err && (
          <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded p-2">{err}</div>
        )}
        <div className="bg-white border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-xs text-slate-500">
              <tr>
                <th className="text-left p-2">Created</th>
                <th className="text-left p-2">Type</th>
                <th className="text-left p-2">Status</th>
                <th className="p-2"></th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 && (
                <tr><td colSpan={4} className="p-4 text-slate-500 text-center">No reports yet.</td></tr>
              )}
              {jobs.map((j) => (
                <tr key={j.id} className="border-t hover:bg-slate-50">
                  <td className="p-2 text-xs text-slate-500">
                    {new Date(j.created_at).toLocaleString()}
                  </td>
                  <td className="p-2 text-xs">
                    {String(j.payload?.report_type ?? j.kind)}
                    {" "}{String(j.payload?.company ?? "")}
                  </td>
                  <td className="p-2">
                    <StatusBadge status={j.status} />
                  </td>
                  <td className="p-2 text-right">
                    <button
                      onClick={() => setOpenJob(j)}
                      className="text-xs text-brand-600 hover:underline"
                    >View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {openJob && (
          <div className="bg-white border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-semibold">Report</h2>
              <button onClick={() => setOpenJob(null)}
                className="text-xs text-slate-500">Close</button>
            </div>
            {openJob.status !== "succeeded" ? (
              <div className="text-sm text-slate-500">
                {openJob.status === "failed"
                  ? `Failed: ${openJob.error}`
                  : "Report still generating…"}
              </div>
            ) : (
              <div className="prose-chat text-slate-800">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {String(openJob.result?.report ?? "")}
                </ReactMarkdown>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls = status === "succeeded" ? "bg-emerald-50 text-emerald-700"
    : status === "failed" ? "bg-rose-50 text-rose-700"
    : status === "running" ? "bg-blue-50 text-blue-700"
    : "bg-slate-100 text-slate-600";
  return <span className={`text-[10px] px-1.5 py-0.5 rounded ${cls}`}>{status}</span>;
}
