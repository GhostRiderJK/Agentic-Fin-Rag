"use client";

import { useEffect, useState } from "react";
import { useApp } from "./AppShell";
import { api, type Workspace } from "@/lib/api";

/** Tiny inline workspace selector. Shown in pages that need a workspace. */
export function WorkspacePicker() {
  const { apiKey, workspaceId, setWorkspaceId } = useApp();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!apiKey) return;
    api.listWorkspaces(apiKey).then(setWorkspaces).catch(() => setWorkspaces([]));
  }, [apiKey]);

  if (!apiKey) return null;

  async function create() {
    if (!apiKey || !name.trim()) return;
    setLoading(true);
    try {
      const ws = await api.createWorkspace(apiKey, name.trim());
      setWorkspaces((w) => [ws, ...w]);
      setWorkspaceId(ws.id);
      setName("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white border rounded-lg p-3 space-y-2">
      <div className="text-xs font-medium text-slate-500">Workspace</div>
      <select
        value={workspaceId ?? ""}
        onChange={(e) => setWorkspaceId(e.target.value || null)}
        className="w-full border rounded-md px-2 py-1.5 text-sm"
      >
        <option value="">— select —</option>
        {workspaces.map((w) => (
          <option key={w.id} value={w.id}>{w.name}</option>
        ))}
      </select>
      <div className="flex gap-1">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New workspace name"
          className="flex-1 border rounded-md px-2 py-1 text-sm"
        />
        <button
          onClick={create}
          disabled={loading || !name.trim()}
          className="bg-brand-600 text-white text-xs px-2 py-1 rounded hover:bg-brand-700 disabled:opacity-50"
        >
          +
        </button>
      </div>
    </div>
  );
}
