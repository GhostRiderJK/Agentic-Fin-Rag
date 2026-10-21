"use client";

import { useState } from "react";
import { useApp } from "./AppShell";
import { api } from "@/lib/api";

// A live tenant key must not live in source: git history keeps every value it
// has ever held. Opt in per machine with NEXT_PUBLIC_DEMO_API_KEY.
const DEMO_API_KEY = process.env.NEXT_PUBLIC_DEMO_API_KEY || "";

/** Inline panel shown when the user isn't connected. Lets them paste an
 * API key (or bootstrap a tenant+user) and pick a workspace. */
export function ConnectionGate({ onConnected }: { onConnected?: () => void }) {
  const { setApiKey } = useApp();
  const [tab, setTab] = useState<"connect" | "bootstrap">("connect");
  const [keyInput, setKeyInput] = useState("");
  const [email, setEmail] = useState("analyst@demo.local");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function connect() {
    setBusy(true);
    setErr(null);
    const key = keyInput.trim();
    if (!key) {
      setErr("Please paste an API key first.");
      setBusy(false);
      return;
    }
    try {
      await api.me(key); // validates the key
      setApiKey(key);
      onConnected?.();
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        setErr(
          "Invalid API key. Get one by running `python -m app.cli seed` in the backend directory, then copy the `api_key` value.",
        );
      } else {
        setErr(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  async function bootstrap() {
    setBusy(true);
    setErr(null);
    try {
      const tenant = await api.createTenant("My Firm");
      const user = await api.createUser(email, "Analyst", tenant.id);
      setApiKey(user.api_key);
      onConnected?.();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-md mx-auto mt-16 bg-white rounded-xl border shadow-sm">
      <div className="flex border-b">
        <button
          onClick={() => setTab("connect")}
          className={`flex-1 py-2 text-sm font-medium ${tab === "connect" ? "border-b-2 border-brand-600 text-brand-600" : "text-slate-500"}`}
        >
          Connect with API key
        </button>
        <button
          onClick={() => setTab("bootstrap")}
          className={`flex-1 py-2 text-sm font-medium ${tab === "bootstrap" ? "border-b-2 border-brand-600 text-brand-600" : "text-slate-500"}`}
        >
          Bootstrap new account
        </button>
      </div>
      <div className="p-4 space-y-3">
        {tab === "connect" ? (
          <>
            <div className="text-xs text-slate-500 space-y-1">
              <div>
                Get your API key from{" "}
                <code className="bg-slate-100 px-1 rounded">python -m app.cli seed</code>
                {" "}in the backend directory:
              </div>
              <pre className="bg-slate-900 text-slate-100 text-[11px] p-2 rounded overflow-x-auto">{`$ python -m app.cli seed
    email   : analyst@demo.local${
      DEMO_API_KEY ? `\n    api_key : ${DEMO_API_KEY}` : ""
    }`}</pre>
              <div>
                {DEMO_API_KEY
                  ? "Copy the api_key value and paste it below."
                  : "The command prints the tenant's api_key; copy it and paste it below. Set NEXT_PUBLIC_DEMO_API_KEY to show it here."}
              </div>
            </div>
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") connect();
              }}
              placeholder="Paste api_key here"
              className="w-full border rounded-md px-3 py-2 text-sm font-mono"
            />
            <button
              onClick={connect}
              disabled={busy}
              className="w-full bg-brand-600 text-white rounded-md py-2 text-sm font-medium hover:bg-brand-700 disabled:opacity-50"
            >
              {busy ? "Connecting…" : "Connect"}
            </button>
          </>
        ) : (
          <>
            <p className="text-xs text-slate-500">
              Creates a brand-new tenant and admin user. Use this for a fresh
              install (no <code className="bg-slate-100 px-1 rounded">ira seed</code> yet).
            </p>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="email"
              className="w-full border rounded-md px-3 py-2 text-sm"
            />
            <button
              onClick={bootstrap}
              disabled={busy || !email.trim()}
              className="w-full bg-brand-600 text-white rounded-md py-2 text-sm font-medium hover:bg-brand-700 disabled:opacity-50"
            >
              {busy ? "Creating…" : "Create tenant + user"}
            </button>
          </>
        )}
        {err && (
          <div className="text-xs text-rose-600 bg-rose-50 p-2 rounded">{err}</div>
        )}
      </div>
    </div>
  );
}
