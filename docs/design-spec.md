# Design Spec — Investment Research Assistant (v2 Architecture)

**Status**: Approved-for-implementation
**Date**: 2026-07-17
**Source**: `prd.md` (sections 1–9 + v2 production architecture)

## 1. Decisions (from user)

| Decision | Choice |
| --- | --- |
| Scope | Full v2 architecture (all 3 PRD phases) |
| LLM | OpenAI default (gpt-4o-mini + text-embedding-3-small); provider-pluggable |
| Infra | Local-first (Qdrant local mode, SQLite, local file storage) |
| UI | Next.js 14 (App Router) + React + TypeScript |
| Secrets | Env-driven; `.env` git-ignored; stub/local providers used when key absent |

## 2. Goals & Success Metrics (from PRD §2)

- 70% faster research
- Citation accuracy >95%
- Response <5s (chat, streaming) — async jobs for thesis/long reports
- Hallucination <2% — measured by the eval harness, not assumed
- Numeric accuracy target added per finance-specific gap

## 3. System Architecture

```
                         User Query
                              │
                              ▼
                   Conversation Manager (memory + query rewrite)
                              │
                              ▼
                   Query Understanding Agent
        (Entity + Intent + Time Range + Metadata Filter)
                              │
            ┌─────────────────┴────────────────────┐
            ▼                                       ▼
  Structured Data Route                   Document Route
  (SQLite financial tables)               (Qdrant vector DB)
            │                                       │
  SQL/Tool Calling                        Hybrid Retrieval
  (Revenue, EPS, PE, …)                   (BM25 + Dense + Metadata)
            │                                       │
            │                                Parent Retrieval
            │                                       │
            ▼                                       ▼
                Re-ranking (Cross-Encoder)
                              │
                              ▼
                  Context Assembly Engine
        (Tables + Text + Conversation History)
                              │
                              ▼
                    Financial Reasoning LLM
                              │
                              ▼
                   Citation Verification (NLI)
                              │
                              ▼
                         Final Response
```

## 4. Repository Layout

```
investment-research/
├── backend/              # FastAPI Python service
│   ├── app/
│   │   ├── main.py                  # ASGI app, router wiring, lifespan
│   │   ├── core/                    # config, security, logging, errors
│   │   ├── providers/               # llm, embeddings, reranker, vectordb
│   │   ├── ingestion/               # parser, ocr, cleaner, tables, chunking
│   │   ├── retrieval/               # query understanding, hybrid, reranker, parent
│   │   ├── reasoning/               # orchestrator + agents + tools
│   │   ├── conversation.py          # session memory, query rewriting
│   │   ├── citation.py              # NLI verification
│   │   ├── evaluation/              # Ragas-style offline harness
│   │   ├── api/                     # FastAPI routers
│   │   ├── db/                      # SQLAlchemy models, migrations
│   │   ├── jobs/                    # in-proc worker (Celery-shaped interface)
│   │   └── seed/                    # sample docs + gold Q&A
│   ├── tests/
│   └── pyproject.toml
├── frontend/             # Next.js 14 + React + TS
├── infra/                # docker-compose.yml — the default way to run, env templates
├── docs/
├── start.bat / stop.bat  # bring the app profile up / stop web + API
└── README.md
```

### 4.1 Run modes

- **Docker (default).** `start.bat`, or
  `docker compose -f infra/docker-compose.yml --profile app up -d`.
  Frontend `127.0.0.1:3000`, API `127.0.0.1:8010` (host port differs from the
  container's 8000 on purpose), MongoDB `127.0.0.1:27019`, Qdrant `6333`.
  Demo credentials come from `docker exec finrag-backend python -m app.cli seed`.
- **Host (no Docker).** `python -m app.cli serve` + `npm run dev`. Separate
  SQLite file and separate MongoDB from the containers — the two modes do not
  share identities, documents, or vectors, so a key created in one is rejected
  by the other.

## 5. Module Responsibilities

Each module is one well-bounded unit:

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `providers/llm` | chat + tool-calling via OpenAI/Anthropic/Ollama/stub | core/config |
| `providers/embeddings` | embed text → vectors | core/config |
| `providers/reranker` | score (query, doc) pairs | providers/llm or local |
| `providers/vectordb` | upsert/search/delete vectors + metadata filtering | qdrant-client |
| `ingestion/*` | PDF/DOCX/HTML/TXT → chunks + tables + embeddings | providers, db |
| `retrieval/*` | query understanding → hybrid search → rerank → parent | providers, db |
| `reasoning/*` | multi-agent orchestration + SQL tools | retrieval, providers |
| `conversation` | session memory + query rewriting for multi-turn | db, providers |
| `citation` | NLI verification of draft vs cited chunks | providers |
| `evaluation` | offline metrics over gold set | retrieval, reasoning |
| `api/*` | thin FastAPI routers: auth, documents, chat, reports, admin, eval | all |
| `core/security` | multi-tenant, RBAC (Admin/Analyst/Viewer), authn | db |
| `jobs/` | async ingestion + long reports (Celery-shaped in-proc) | ingestion, reasoning |
| `db/` | SQLAlchemy models + migrations (SQLite default, Postgres-ready) | — |

## 6. Data Model (relational)

Tenants, Users (roles), Workspaces, Documents (status lifecycle), Chunks (parent/child, metadata), StructuredFinancials (company, metric, period, value, unit), Citations, Conversations, Messages, Jobs, EvalRuns, EvalItems.

Document lifecycle: `active | superseded | archived`. Filter defaults to most-recent active unless requested.

## 7. Ingestion Pipeline

`upload → parser → OCR (if scanned) → cleaner (incl. glossary mapping) → table extractor → metadata extractor → parent/child chunker → embeddings → vector DB upsert + structured DB insert`.

- Layout-aware parsing via `unstructured`/`pymupdf` with Docling-aware table mode.
- Tables extracted to Markdown + JSON rows; numeric rows also land in `StructuredFinancials`.
- Chunking: section-aware → paragraph (parent) → sentence (child). Search on child, expand to parent for the LLM.

## 8. Retrieval Pipeline

`question → query understanding (company/year/quarter/section + intent) → metadata prefilter → hybrid (BM25 over chunks + dense over Qdrant) → reranker (top-k) → parent expansion → context assembly`.

## 9. Reasoning / Multi-Agent

- **Retriever Agent** — runs the retrieval pipeline.
- **Analyst Agent** — composes the answer; may call SQL tools for numeric facts.
- **Citation Agent** — NLI-verifies each cited claim against its chunk; rejects unsupported.
- **Formatter Agent** — formats final response with bullet summary + citations.

## 10. Evaluation Harness

Metrics over gold set: Faithfulness, Context Precision, Context Recall, Answer Relevance, Citation Precision, Numeric Accuracy. Stored in `EvalRuns`/`EvalItems`; runnable via CLI and admin API.

## 11. Non-Functional

- Latency: chat <5s (streaming); thesis/long via async job with completion notification.
- Capacity: 1000+ PDFs per workspace (clarified per-workspace, not global).
- Concurrency: async FastAPI + worker pool.
- Security: tenant isolation enforced at repository layer; API keys/JWT auth; RBAC.

## 12. Implementation Sequence

1. Repo + config + providers + db models + security base
2. Ingestion
3. Retrieval
4. Reasoning + tools
5. Citation + conversation + evaluation
6. API + jobs
7. Frontend
8. Seed data + README + demo
9. End-to-end verification

## 13. Self-Review Notes (resolved inline)

- No placeholders — all sections concrete.
- Capacity clarified per-workspace (resolves PRD NFR contradiction).
- Latency split chat vs. long reports (resolves PRD NFR contradiction).
- Hallucination metric made measurable via eval harness (resolves PRD float metric).
- Citation correctness enforced via NLI, not just format (resolves PRD §9 critique).
