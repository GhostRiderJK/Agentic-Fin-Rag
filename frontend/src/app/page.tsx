"use client";

import { useEffect, useRef, useState } from "react";
import { useApp } from "@/components/AppShell";
import { ConnectionGate } from "@/components/ConnectionGate";
import { WorkspacePicker } from "@/components/WorkspacePicker";
import { Answer } from "@/components/Answer";
import { api, type ChatResponse } from "@/lib/api";

interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  resp?: ChatResponse;
}

export default function ChatPage() {
  const { apiKey, workspaceId } = useApp();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  if (!apiKey) return <ConnectionGate />;

  async function send() {
    if (!apiKey || !workspaceId || !input.trim() || busy) return;
    const q = input.trim();
    setInput("");
    setBusy(true);
    setErr(null);
    const userTurn: Turn = { id: crypto.randomUUID(), role: "user", content: q };
    setTurns((t) => [...t, userTurn]);
    try {
      const resp = await api.chat(apiKey, q, workspaceId, conversationId ?? undefined);
      if (resp.conversation_id) setConversationId(resp.conversation_id);
      setTurns((t) => [...t, {
        id: resp.message_id || crypto.randomUUID(),
        role: "assistant", content: resp.answer, resp,
      }]);
    } catch (e) {
      setErr((e as Error).message);
      setTurns((t) => [...t, {
        id: crypto.randomUUID(), role: "assistant",
        content: `Error: ${(e as Error).message}`,
      }]);
    } finally {
      setBusy(false);
    }
  }

  const suggestions = [
    "What was ACME's net income in fiscal year 2025?",
    "What are the main risks ACME faces?",
    "Compare ACME's gross margin 2024 vs 2025",
    "What is driving ACME's Services growth?",
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 py-4 grid grid-cols-12 gap-4 h-[calc(100vh-3.5rem)]">
      <aside className="col-span-3 space-y-3 overflow-y-auto">
        <WorkspacePicker />
        <div className="bg-white border rounded-lg p-3">
          <div className="text-xs font-medium text-slate-500 mb-2">Try asking</div>
          <ul className="space-y-1">
            {suggestions.map((s) => (
              <li key={s}>
                <button
                  onClick={() => setInput(s)}
                  className="text-xs text-left text-slate-700 hover:text-brand-600 w-full"
                >
                  {s}
                </button>
              </li>
            ))}
          </ul>
        </div>
        {!workspaceId && (
          <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
            Pick or create a workspace to start chatting.
          </div>
        )}
      </aside>

      <section className="col-span-9 flex flex-col bg-white border rounded-lg overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {turns.length === 0 && (
            <div className="h-full grid place-items-center text-center">
              <div className="max-w-md text-slate-500">
                <h2 className="text-lg font-semibold text-slate-700">
                  Ask anything about your documents
                </h2>
                <p className="text-sm mt-1">
                  Upload annual reports, earnings transcripts, and 10-Ks, then
                  ask questions with citations and verified answers.
                </p>
              </div>
            </div>
          )}
          {turns.map((t) => (
            <div key={t.id} className={t.role === "user" ? "flex justify-end" : ""}>
              <div className={`max-w-[85%] rounded-lg px-4 py-3 ${
                t.role === "user"
                  ? "bg-brand-600 text-white"
                  : "bg-slate-50 border"
              }`}>
                {t.role === "user" ? (
                  <div className="text-sm whitespace-pre-wrap">{t.content}</div>
                ) : t.resp ? (
                  <Answer resp={t.resp} />
                ) : (
                  <div className="text-sm whitespace-pre-wrap">{t.content}</div>
                )}
              </div>
            </div>
          ))}
          {busy && (
            <div className="flex">
              <div className="bg-slate-50 border rounded-lg px-4 py-3 text-sm text-slate-500">
                <span className="inline-block animate-pulse">Thinking…</span>
              </div>
            </div>
          )}
          {err && (
            <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded p-2">
              {err}
            </div>
          )}
          <div ref={endRef} />
        </div>
        <div className="border-t p-3">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder={
                workspaceId
                  ? "Ask a question (Enter to send, Shift+Enter for newline)"
                  : "Select a workspace first…"
              }
              disabled={!workspaceId}
              rows={2}
              className="flex-1 border rounded-md px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
            <button
              onClick={send}
              disabled={!workspaceId || busy || !input.trim()}
              className="bg-brand-600 text-white px-4 rounded-md text-sm font-medium hover:bg-brand-700 disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
