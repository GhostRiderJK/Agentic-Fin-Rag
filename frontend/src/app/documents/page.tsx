"use client";

import { useEffect, useRef, useState } from "react";
import { useApp } from "@/components/AppShell";
import { ConnectionGate } from "@/components/ConnectionGate";
import { WorkspacePicker } from "@/components/WorkspacePicker";
import { api, type Document } from "@/lib/api";

export default function DocumentsPage() {
  const { apiKey, workspaceId } = useApp();
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [form, setForm] = useState({
    company: "", ticker: "", year: "", quarter: "", doc_type: "annual_report",
  });
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!apiKey || !workspaceId) return;
    setLoading(true);
    api.listDocuments(apiKey, workspaceId)
      .then(setDocs)
      .catch((e) => setErr((e as Error).message))
      .finally(() => setLoading(false));
  }, [apiKey, workspaceId]);

  if (!apiKey) return <ConnectionGate />;

  async function upload() {
    if (!apiKey || !workspaceId || !fileRef.current?.files?.[0]) return;
    setBusy(true); setErr(null);
    try {
      const file = fileRef.current.files[0];
      await api.uploadDocument(apiKey, workspaceId, file, {
        company: form.company,
        ticker: form.ticker,
        year: form.year,
        quarter: form.quarter,
        doc_type: form.doc_type,
        async_ingest: "true",
      });
      // Refresh list (ingest is async; show whatever's there now)
      setTimeout(async () => {
        const d = await api.listDocuments(apiKey, workspaceId);
        setDocs(d);
      }, 1500);
      if (fileRef.current) fileRef.current.value = "";
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    if (!apiKey || !workspaceId) return;
    if (!confirm("Delete this document and all its chunks?")) return;
    try {
      await api.deleteDocument(apiKey, workspaceId, id);
      setDocs((d) => d.filter((x) => x.id !== id));
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-12 gap-4">
      <aside className="col-span-3 space-y-3">
        <WorkspacePicker />
      </aside>
      <section className="col-span-9 space-y-4">
        <h1 className="text-xl font-semibold">Documents</h1>

        <div className="bg-white border rounded-lg p-4 space-y-3">
          <div className="font-medium text-sm">Upload document</div>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.html,.htm,.txt,.md"
            className="block text-sm"
          />
          <div className="grid grid-cols-2 gap-2">
            <Field label="Company" value={form.company}
              onChange={(v) => setForm({ ...form, company: v })} />
            <Field label="Ticker" value={form.ticker}
              onChange={(v) => setForm({ ...form, ticker: v })} />
            <Field label="Year" value={form.year}
              onChange={(v) => setForm({ ...form, year: v })} />
            <Field label="Quarter" value={form.quarter}
              onChange={(v) => setForm({ ...form, quarter: v })} />
            <div>
              <label className="text-xs text-slate-500">Document type</label>
              <select
                value={form.doc_type}
                onChange={(e) => setForm({ ...form, doc_type: e.target.value })}
                className="w-full border rounded-md px-2 py-1.5 text-sm"
              >
                {["annual_report", "quarterly_report", "earnings_call",
                  "investor_presentation", "sustainability_report", "news",
                  "research_note", "other"].map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>
          <button
            onClick={upload}
            disabled={busy || !workspaceId}
            className="bg-brand-600 text-white text-sm px-3 py-1.5 rounded hover:bg-brand-700 disabled:opacity-50"
          >
            {busy ? "Uploading…" : "Upload"}
          </button>
          {!workspaceId && (
            <span className="text-xs text-amber-700 ml-2">Pick a workspace first.</span>
          )}
        </div>

        {err && (
          <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded p-2">
            {err}
          </div>
        )}

        <div className="bg-white border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-xs text-slate-500">
              <tr>
                <th className="text-left p-2">Title</th>
                <th className="text-left p-2">Company</th>
                <th className="text-left p-2">Type</th>
                <th className="text-left p-2">Year</th>
                <th className="text-left p-2">Status</th>
                <th className="text-left p-2">Tables / OCR</th>
                <th className="p-2"></th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr><td colSpan={7} className="p-4 text-slate-500 text-center">Loading…</td></tr>
              )}
              {!loading && docs.length === 0 && (
                <tr><td colSpan={7} className="p-4 text-slate-500 text-center">No documents yet.</td></tr>
              )}
              {docs.map((d) => (
                <tr key={d.id} className="border-t hover:bg-slate-50">
                  <td className="p-2">
                    <div className="font-medium">{d.title}</div>
                    <div className="text-[11px] text-slate-500">{d.filename}</div>
                  </td>
                  <td className="p-2">{d.company || "—"}</td>
                  <td className="p-2 text-xs">{d.doc_type}</td>
                  <td className="p-2">{d.year || "—"}</td>
                  <td className="p-2">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      d.status === "active" ? "bg-emerald-50 text-emerald-700"
                      : d.status === "superseded" ? "bg-amber-50 text-amber-700"
                      : "bg-slate-100 text-slate-600"
                    }`}>{d.status}</span>
                  </td>
                  <td className="p-2 text-xs text-slate-500">
                    {String(d.parse_meta?.n_tables ?? 0)} tables · {" "}
                    {String(d.parse_meta?.needs_ocr ?? false) === "true" ? "OCR" : "text"}
                  </td>
                  <td className="p-2 text-right">
                    <button
                      onClick={() => remove(d.id)}
                      className="text-xs text-rose-500 hover:text-rose-700"
                    >Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Field({ label, value, onChange }: {
  label: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="text-xs text-slate-500">{label}</label>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full border rounded-md px-2 py-1.5 text-sm"
      />
    </div>
  );
}
