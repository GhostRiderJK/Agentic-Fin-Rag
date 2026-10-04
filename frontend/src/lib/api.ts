// Thin typed client for the FastAPI backend. All requests go through the
// Next.js /api/* rewrite (see next.config.js) so the browser stays same-origin
// and CORS isn't a concern in dev.

export type Role = "admin" | "analyst" | "viewer";
export type DocumentStatus = "active" | "superseded" | "archived";
export type DocumentType =
  | "annual_report" | "quarterly_report" | "earnings_call"
  | "investor_presentation" | "sustainability_report" | "news"
  | "research_note" | "other";

export interface User {
  id: string; email: string; name: string; role: Role;
  tenant_id: string; api_key: string; is_active: boolean;
}

export interface Workspace {
  id: string; name: string; description: string; created_at: string;
}

export interface Document {
  id: string; workspace_id: string; title: string; filename: string;
  doc_type: DocumentType; company: string; ticker: string;
  year: number | null; quarter: string; section: string;
  status: DocumentStatus; parse_meta: Record<string, unknown>;
  created_at: string;
}

export interface Citation {
  index: number; chunk_id: string; document_id: string;
  company: string | null; year: number | null; quarter: string | null;
  doc_type: string | null; section: string | null; page: number;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  plan: Record<string, unknown>;
  tool_results: Record<string, unknown>[];
  verification: Record<string, unknown>;
  retrieval: Record<string, unknown>;
  latency_ms: number;
  notes: string[];
  message_id?: string | null;
  conversation_id?: string | null;
}

export interface Message {
  id: string; role: string; content: string;
  citations: unknown[]; created_at: string; feedback: number | null;
}

export interface Job {
  id: string; kind: string; status: string; progress: number;
  error: string; created_at: string; started_at: string | null;
  finished_at: string | null;
  payload?: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface EvalRun {
  id: string; suite: string; n_items: number;
  metrics: Record<string, number>; created_at: string;
}

const API_BASE = "/api";

function headers(apiKey: string | null): HeadersInit {
  const h: Record<string, string> = {};
  if (apiKey) h["X-API-Key"] = apiKey;
  h["Content-Type"] = "application/json";
  return h;
}

async function request<T>(
  path: string,
  opts: RequestInit & { apiKey?: string | null } = {},
): Promise<T> {
  const { apiKey, ...init } = opts;
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...headers(apiKey ?? null), ...(init.headers as object) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      detail = j.message || JSON.stringify(j);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  // Auth
  createTenant: (name: string) =>
    request<Workspace & { id: string; name: string }>("/auth/tenants", {
      method: "POST", body: JSON.stringify({ name }),
    }),
  createUser: (email: string, name: string, tenantId: string, apiKey?: string | null) =>
    request<User>("/auth/users", {
      method: "POST",
      body: JSON.stringify({ email, name, tenant_id: tenantId, role: "admin" }),
      headers: apiKey ? { "X-API-Key": apiKey } : undefined,
    }),
  me: (apiKey: string) => request<User>("/auth/me", { apiKey }),

  // Workspaces
  listWorkspaces: (apiKey: string) =>
    request<Workspace[]>("/workspaces", { apiKey }),
  createWorkspace: (apiKey: string, name: string, description = "") =>
    request<Workspace>("/workspaces", {
      method: "POST", apiKey,
      body: JSON.stringify({ name, description }),
    }),

  // Documents
  listDocuments: (apiKey: string, wsId: string) =>
    request<Document[]>(`/workspaces/${wsId}/documents`, { apiKey }),
  uploadDocument: async (
    apiKey: string, wsId: string,
    file: File, fields: Record<string, string>,
  ) => {
    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    const res = await fetch(`${API_BASE}/workspaces/${wsId}/documents`, {
      method: "POST",
      headers: { "X-API-Key": apiKey },
      body: fd,
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.message || `${res.status}`);
    }
    return res.json();
  },
  deleteDocument: (apiKey: string, wsId: string, docId: string) =>
    request<{ deleted: string }>(
      `/workspaces/${wsId}/documents/${docId}`,
      { method: "DELETE", apiKey },
    ),

  // Chat
  chat: (apiKey: string, question: string, wsId: string,
         conversationId?: string) =>
    request<ChatResponse>("/chat", {
      method: "POST", apiKey,
      body: JSON.stringify({
        question, workspace_id: wsId, conversation_id: conversationId ?? null,
      }),
    }),
  listConversations: (apiKey: string, wsId: string) =>
    request<{ id: string; title: string; workspace_id: string;
             created_at: string; updated_at: string }[]>(
      `/workspaces/${wsId}/conversations`, { apiKey },
    ),
  listMessages: (apiKey: string, wsId: string, convId: string) =>
    request<Message[]>(`/workspaces/${wsId}/conversations/${convId}/messages`,
                       { apiKey }),

  // Reports
  createReport: (apiKey: string, wsId: string, company: string,
                 reportType: "summary" | "thesis", year?: number) =>
    request<Job>("/reports", {
      method: "POST", apiKey,
      body: JSON.stringify({
        workspace_id: wsId, company,
        report_type: reportType, year,
      }),
    }),
  listJobs: (apiKey: string, wsId?: string) =>
    request<Job[]>(`/jobs${wsId ? `?workspace_id=${wsId}` : ""}`, { apiKey }),

  // Eval
  runEval: (apiKey: string, wsId: string) =>
    request<EvalRun>("/admin/eval/run", {
      method: "POST", apiKey,
      body: JSON.stringify({ workspace_id: wsId }),
    }),
  listEvalRuns: (apiKey: string) =>
    request<EvalRun[]>("/admin/eval/runs", { apiKey }),
};
