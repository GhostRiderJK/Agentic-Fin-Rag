"use client";

import { createContext, useContext, useEffect, useState } from "react";

const LS_KEY = "ira.apiKey";
const LS_WS = "ira.workspaceId";

interface AppCtx {
  apiKey: string | null;
  setApiKey: (k: string | null) => void;
  workspaceId: string | null;
  setWorkspaceId: (id: string | null) => void;
}

const Ctx = createContext<AppCtx>(null as unknown as AppCtx);

export function AppShell({ children }: { children: React.ReactNode }) {
  const [apiKey, setApiKeyState] = useState<string | null>(null);
  const [workspaceId, setWorkspaceIdState] = useState<string | null>(null);

  useEffect(() => {
    const k = localStorage.getItem(LS_KEY);
    const w = localStorage.getItem(LS_WS);
    if (k) setApiKeyState(k);
    if (w) setWorkspaceIdState(w);
  }, []);

  const setApiKey = (k: string | null) => {
    setApiKeyState(k);
    if (k) localStorage.setItem(LS_KEY, k);
    else localStorage.removeItem(LS_KEY);
  };
  const setWorkspaceId = (id: string | null) => {
    setWorkspaceIdState(id);
    if (id) localStorage.setItem(LS_WS, id);
    else localStorage.removeItem(LS_WS);
  };

  return (
    <Ctx.Provider value={{ apiKey, setApiKey, workspaceId, setWorkspaceId }}>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1">{children}</main>
      </div>
    </Ctx.Provider>
  );
}

export function useApp() {
  return useContext(Ctx);
}

function Header() {
  const { apiKey, setApiKey } = useApp();
  return (
    <header className="border-b bg-white">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-4">
        <a href="/" className="flex items-center gap-2">
          <div>
            <div className="font-semibold text-sm leading-tight">
              Investment Research Assistant
            </div>
            <div className="text-[11px] text-slate-500 leading-tight">
              Agentic Financial Intelligence Platform
            </div>
          </div>
        </a>
        <nav className="ml-6 flex items-center gap-4 text-sm">
          <a href="/" className="text-slate-700 hover:text-brand-600">Chat</a>
          <a href="/documents" className="text-slate-700 hover:text-brand-600">Documents</a>
          <a href="/reports" className="text-slate-700 hover:text-brand-600">Reports</a>
          <a href="/eval" className="text-slate-700 hover:text-brand-600">Eval</a>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          {apiKey ? (
            <>
              <span className="text-xs px-2 py-1 bg-emerald-50 text-emerald-700 rounded">
                Connected
              </span>
              <button
                onClick={() => setApiKey(null)}
                className="text-xs text-slate-500 hover:text-rose-600"
              >
                Disconnect
              </button>
            </>
          ) : (
            <span className="text-xs px-2 py-1 bg-amber-50 text-amber-700 rounded">
              Not connected
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
