# Investment Research Assistant

An **Agentic Financial Intelligence Platform** for equity research analysts — ask questions in natural language and get answers grounded in your uploaded annual reports, 10-Ks, earnings transcripts, and investor presentations, with verified citations and supporting tables.

Built against the spec in [`prd.md`](./prd.md). [`docs/STATUS.md`](./docs/STATUS.md) is the authoritative record of which parts are verified, which are stubbed, and which are still open — read it before trusting any claim on this page.

---

## What it does

- **Dual knowledge system** — narrative text → vector DB (Qdrant); numeric facts → structured store (MongoDB `structured_financials`). The LLM calls tools against it instead of reciting numbers. *Measured caveat:* on the real PIF corpus this table holds **0 rows** although the parser extracted 18 tables, so metric-route questions currently get nothing from it.
- **Layout-aware ingestion** — PDF/DOCX/HTML/TXT → text + tables + parent/child chunks + metadata. Scanned PDFs (under 50 chars/page) fall back to Tesseract OCR; the OCR path returns **no tables**, so a scanned report yields prose but no structured numbers.
- **Hybrid retrieval** — dense (Qdrant, `nomic-embed-text` 768-d) + per-query BM25Okapi + metadata prefilter, fused as `0.6·dense + 0.4·BM25 + 0.25·query-coverage`, then a rerank stage, then parent expansion into a budgeted context (8 chunks / 24,000 chars). **The reranker in use is a query-term-coverage stub** — the Cohere and `cross-encoder/ms-marco-MiniLM-L-6-v2` classes exist in `providers/reranker.py`, but `sentence-transformers` is not a dependency, so no neural reranker runs.
- **Page-diversity selection** — optional re-selection of the returned set to at most one (or two) chunks per `(document_id, physical_page)`. Measured on the PIF corpus: source+page recall at k=10 5/10 → 7/10 and mean unique pages 3.6 → 9.2 with zero per-question regressions. **Implemented but off by default** (`RETRIEVAL_PAGE_QUOTA=0`).
- **Query understanding** — extracts company / year / quarter / doc-type / metric-intent and builds a metadata filter automatically. *Known correctness risk:* the company filter is silently dropped when it matches nothing, and an unrecognised year drops the year filter, so a missing entity can be answered from the wrong document.
- **Agentic orchestration** — one deterministic orchestrator plays Retriever → Analyst (bounded 3-turn tool loop) → Citation Verifier → Formatter. There is no LLM planner, by design.
- **Citation verification** — claims are split and checked against cited chunks, with numeric-subset, currency and scale binding; over 50% unsupported fails closed into an abstention rather than a printed answer.
- **Conversation memory** — multi-turn follow-ups resolve pronouns and references.
- **Reports** — async earnings-summary and investment-thesis generation.
- **Evaluation harness** — runs a gold Q&A suite and reports faithfulness, answer relevance, context precision/recall, citation precision and numeric accuracy. **Three of those are not real measurements**: `context_precision` and `context_recall` are the same placeholder `min(1.0, n_retrieved/5)`, and `citation_precision` only checks that the answer contains a digit. Real page-level recall comes from the stage-attribution harness (`backend/scripts/exp_pif_stages.py`), not from this dashboard.
- **Multi-tenant + RBAC** — tenant isolation at the repository layer, workspaces, admin/analyst/viewer roles. Four routes still lack tenant checks and are pinned as strict xfails (see *Security notes*).
- **Document lifecycle** — active / superseded / archived.

## Screenshots — proof it works

Captured live from the running app on 2026-10-16 (GPT-4o-mini + real EDGAR 10-K/10-Q
filings from AAPL, AMZN, GOOGL, JPM, JNJ, INTC, DIS, CSCO, CRM, ADBE, ingested into the
Demo Workspace, alongside the two seeded ACME demo documents).
The full-resolution originals live in [`docs/screenshots/`](./docs/screenshots).

| | | |
|:---:|:---:|:---:|
| **Connect** — paste an API key, pick a workspace | **Chat — grounded answer** with citations + `Verified` badge | **Chat — citation guardrail** catches an unsupported number |
| ![](./docs/screenshots/01-connect.png) | ![](./docs/screenshots/02-chat-answer.png) | ![](./docs/screenshots/03-chat-citations.png) |
| **Documents** — real SEC filings, layout-aware ingestion, lifecycle states | **Eval dashboard** — faithfulness, retrieval, citation & numeric metrics | |
| ![](./docs/screenshots/04-documents.png) | ![](./docs/screenshots/05-eval.png) | |

A few details worth noticing in those shots:

- The **risks** answer comes back structured, with five supporting sources and a
  green `Verified` badge — every claim passed citation verification.
- The **net income** answer is flagged `Rejected` because the model rounded
  $25B → $28B; the citation verifier caught the mismatch rather than letting an
  unsupported number through. That guardrail working is the point.
- The **Documents** table shows ~50 real SEC filings with `active` lifecycle
  states, parsed company/ticker/year metadata, doc-type detection
  (`annual_report`, `earnings_call`, `10-K`, `10-Q`, …) and per-document table
  counts (38–283).
- The **Eval** dashboard shows faithfulness 0.71, answer relevance 0.54 and
  numeric accuracy 0.57 on the 7-item seeded suite — and **context precision
  1.00, context recall 1.00, citation precision 1.00, which mean nothing**: those
  three are placeholder formulas (see the evaluation bullet above). Read the other
  three, and read [`docs/STATUS.md`](./docs/STATUS.md) before believing any of them.
- Timestamps are stored as **naive UTC** and rendered without conversion, so the
  eval history can read several hours off the wall clock. (The `7/17/2026` on that
  screenshot is a capture made on 2026-10-16; the ~5.5 h naive-UTC defect does not
  account for a three-month gap, so that particular value is unexplained and is
  left as an open question rather than given a cause.)

## Architecture

See [`docs/design-spec.md`](./docs/design-spec.md) for the full design.

```
                         User Query
                              │
                              ▼
                   Conversation Manager (memory + rewrite)
                              │
                              ▼
                   Query Understanding Agent
        (Entity + Intent + Time Range + Metadata Filter)
                              │
            ┌─────────────────┴────────────────────┐
            ▼                                       ▼
  Structured Data Route                   Document Route
  (MongoDB structured_financials)         (Qdrant, 768-d nomic-embed-text)
            │                                       │
  4 tools: get_metric / list_metrics        Hybrid retrieval
  compare_metrics / list_periods           (dense 0.6 + BM25 0.4 + coverage 0.25,
            │                               metadata prefilter, pool 30)
            │                                       │
            │                                       ▼
            │                          Rerank — coverage STUB by default
            │                                       │
            │                                       ▼
            │                          Page-diversity select (off by default)
            │                                       │
            └───────────────────┬───────────────────┘
                                ▼
                    Context assembly (8 chunks / 24,000 chars,
                    parent expansion, [n] markers)
                                │
                                ▼
                    Financial Reasoning LLM (≤3 tool turns)
                                │
                                ▼
              Citation verification (claim split, number /
                 currency / scale binding, fails closed)
                                │
                                ▼
                    Final response  or  abstention
```

## Tech stack

| Component | Choice | Notes |
| --- | --- | --- |
| LLM | `stub` by default; `omniroute` (OpenAI-compatible gateway) in this deployment | Also `openai` / `anthropic` / `ollama`. **`AnthropicLLM` and `OllamaLLM` never send `tools`**, so tool-calling is silently unavailable on them. |
| Embeddings | `stub` by default; **`ollama` / `nomic-embed-text`, 768-d** in this deployment | `openai` path exists. The gateway offers **no** embedding or rerank route (all 8 candidates 404), and `sentence-transformers` is not a dependency. |
| Vector DB | Qdrant v1.11 (local mode default; remote in Docker) | Atlas `$vectorSearch` was spiked (12/12) and is **not wired into the app**. |
| Document store | **MongoDB** (`documents`, `chunks`, `structured_financials` + 10 more collections) | Migrated for document/chunk/financial routes. |
| Identity store | SQLAlchemy + SQLite (`tenants`, `users`, `workspaces`, `conversations`, `messages`, `jobs`, `eval_*`, `feedback_logs`) | **Still SQL** — workspaces and users are not yet on Mongo; Postgres-switchable in principle. |
| Reranker | `stub` (query-term coverage) | `cohere` and `cross_encoder` classes exist; neither runs here. |
| Document parsing | pdfplumber, python-docx, BeautifulSoup, Tesseract OCR | Printed-page footer recovery from `page.chars`, with a glyph-doubling fix. |
| Frontend | Next.js 14 (App Router) + React + TypeScript + Tailwind | `experimental.proxyTimeout` raised to 900 s so long chats survive the `/api` proxy. |
| Jobs | In-process worker (Celery-shaped interface) | ingest / report / eval are queued; eval returns `202` + `job_id`. |
| Tests | 1017 collected / 1006 passed / 11 xfailed (25 files) | Mongo suites are parameterised `mock` + `real` and **fail loudly** when `:27018` is unreachable. |

## Quick start

Two ways to run it. Pick one — **the two do not share a database**, so a key
created by one will not authenticate the other.

### A. Docker (how this repo runs today)

`start.bat` and `stop.bat` in the repo root wrap the compose stack: they check
the daemon, bring up data services + API + web, then wait for the API health
check before printing the URLs. The raw commands are:

```powershell
# from the repo root
docker compose -f infra/docker-compose.yml --profile app up -d                     # start
docker compose -f infra/docker-compose.yml --profile app stop backend frontend     # stop
```

Published ports (all bound to `127.0.0.1`, all host-overrideable):

| What | Address | Env override |
| --- | --- | --- |
| Frontend | http://127.0.0.1:3000 | `FRONTEND_HOST_PORT` |
| API (`/docs` too) | http://127.0.0.1:8010 | `BACKEND_HOST_PORT` |
| MongoDB (Atlas-local) | `127.0.0.1:27019` | `MONGO_HOST_PORT` |
| MongoDB **test** instance (volume-free, so tests can never touch app data) | `127.0.0.1:27018` | `MONGO_TEST_PORT` |
| Qdrant | http://127.0.0.1:6333 | — |

> The API is published on **8010**, not 8000: the compose file deliberately keeps
> the host's 8000 free so a host-side `ira serve` can coexist. The web container
> reaches the API over the compose network at `backend:8000` regardless of this
> mapping.

**Getting the API key — run the CLI inside the container.** The container's
users/workspaces live in the `backend_data` volume (`/app/data/app.db`) and its
documents in MongoDB on 27019, so the key has to be created by the container:

```powershell
docker exec finrag-backend python -m app.cli seed
# ✓ created tenant: …
#     email   : analyst@demo.local
#     api_key : 0123456789abcdef…     <-- paste this into the UI
#     workspace: …  (Demo Workspace)
```

Re-running it is safe: it reuses the existing tenant/user/workspace and still
prints the `api_key`, so it doubles as credential recovery.

Every other CLI command runs the same way, against the container's own config:

```powershell
docker exec finrag-backend python -m app.cli --help
docker exec finrag-backend python -m app.seed.make_samples --out ./data/samples
docker logs --tail 50 finrag-backend        # when something looks wrong
```

Then open http://127.0.0.1:3000 and follow the **In the UI** steps at the end of
this section.

### B. On the host, without Docker

#### 1. Backend

```bash
cd backend
python -m pip install -e ".[dev]"

# Configure for LIVE OpenAI (gpt-4o-mini + text-embedding-3-small).
# Copy the env template and edit it: set OPENAI_API_KEY and switch the
# providers to `openai`:
cp ../.env.example ../.env
# edit ../.env:
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
#   EMBEDDINGS_PROVIDER=openai
#
# Without these the system still runs in OFFLINE stub mode (no API spend,
# deterministic placeholders) — useful for tests and demos without a key.

python -m app.cli init-db
python -m app.cli seed                       # creates demo tenant/user/workspace
python -m app.seed.make_samples --out ./data/samples
```

#### 2. Run live (important: ingest via the server, not the CLI)

> ⚠️ **Why this matters:** Qdrant local mode uses a single-writer storage
> lock. If you ingest from a separate CLI process while the server runs,
> the server can't see those vectors and retrieval returns zero hits.
> Always ingest **after** starting the server, via its HTTP API (the UI
> and the snippet below do this correctly).

```bash
# Start the API server first
python -m app.cli serve                       # → http://127.0.0.1:8000  (docs at /docs)

# In another terminal: ingest documents THROUGH the running server so
# vectors live in the server's process
API_KEY=$(python -c "from app.db.base import db_scope; from sqlalchemy import select; from app.db.models import User
import sys
with db_scope() as db: print(db.scalar(select(User).where(User.email=='analyst@demo.local')).api_key)")
WS=$(python -c "from app.db.base import db_scope; from sqlalchemy import select; from app.db.models import Workspace
with db_scope() as db: print(db.scalar(select(Workspace).order_by(Workspace.created_at.desc())).id)")

curl -X POST -H "X-API-Key: $API_KEY" \
  -F "file=@./data/samples/acme_2025_annual_report.txt" \
  -F "company=ACME" -F "ticker=ACME" -F "year=2025" -F "async_ingest=false" \
  "http://127.0.0.1:8000/workspaces/$WS/documents"

curl -X POST -H "X-API-Key: $API_KEY" \
  -F "file=@./data/samples/acme_2025_q4_earnings_call.txt" \
  -F "company=ACME" -F "ticker=ACME" -F "year=2025" -F "quarter=Q4" -F "async_ingest=false" \
  "http://127.0.0.1:8000/workspaces/$WS/documents"

# Seed structured financials (writes only to the relational DB; safe cross-process)
python -m app.seed.seed_financials

# Ask a question via the live server (GPT-4o-mini + tool-calling + citations)
curl -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d "{\"question\":\"What are ACME's main risks?\",\"workspace_id\":\"$WS\"}" \
  http://127.0.0.1:8000/chat
```

For a fully offline / no-key run, the same commands work — the answers
will be deterministic stub placeholders instead of GPT-4o-mini output,
but retrieval, citations, and the eval harness all run unchanged.


#### 3. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

> If the Docker web container is running, it already holds `127.0.0.1:3000`.
> Stop it first (`docker compose -f infra/docker-compose.yml --profile app stop frontend`)
> or run the dev server on another port (`npm run dev -- -p 3001`).

#### In the UI

1. Click **Connect with API key** and paste the `api_key` printed by `ira seed`
   — run it inside `finrag-backend` for the Docker app, or in `backend/` for a
   host run — or click **Bootstrap new account** to create a fresh tenant + admin user.
2. Create or pick a **workspace**.
3. Go to **Documents** → upload a PDF/DOCX/TXT, or run the demo.
4. Go to **Chat** → ask questions; answers come with citations + verification badges.
5. Go to **Reports** → generate an earnings summary or investment thesis.
6. Go to **Eval** → run the gold Q&A suite and inspect metrics.

## CLI commands

```
ira serve            Run the FastAPI server (uvicorn)
ira init-db          Create database tables
ira seed             Create demo tenant + user + workspace
ira ingest <file>    Ingest a document synchronously
ira eval             Run the evaluation suite
ira ask "<question>" One-shot Q&A
```

Run any with `python -m app.cli <command>` from the `backend/` directory — or,
when the app runs in Docker, with
`docker exec finrag-backend python -m app.cli <command>` so it uses the
container's database and config instead of the host's.

## API overview

All endpoints are documented at `/docs` (Swagger). Auth via `X-API-Key` header or `Authorization: Bearer <jwt>`.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/auth/tenants` | Create tenant |
| POST | `/auth/users` | Create user (bootstrap-friendly) |
| POST | `/auth/login` | Email/password → JWT |
| GET  | `/auth/me` | Current user |
| POST/GET | `/workspaces` | Create/list workspaces |
| POST | `/workspaces/{id}/documents` | Upload + async-ingest |
| GET/PATCH/DELETE | `/workspaces/{id}/documents/{id}` | Document lifecycle |
| POST | `/chat` | Ask a question (citations + verification) |
| POST | `/chat/stream` | SSE streaming chat |
| GET | `/workspaces/{id}/conversations` | Conversation history |
| POST | `/messages/{id}/feedback` | Thumbs up/down |
| POST | `/reports` | Async earnings summary / thesis |
| GET | `/jobs`, `/jobs/{id}` | Job status |
| POST | `/admin/eval/run` | Queue an eval suite → `202` + job id; poll `/jobs/{id}` |
| GET | `/admin/eval/runs` | Past eval runs |
| GET | `/admin/eval/suites` | Gold sets available, by name |

## Configuration

All config is env-driven (see [`.env.example`](./.env.example)). Highlights:

| Var | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `stub` | `openai` / `anthropic` / `ollama` / `stub` / `omniroute` |
| `LLM_BASE_URL` / `LLM_API_KEY` | gateway at `http://127.0.0.1:20128/v1` | Aliases win over `OMNIROUTE_BASE_URL` / `OMNIROUTE_API_KEY`, which also work. |
| `CHAT_MODEL` / `REASONING_MODEL` | `nvidia/nvidia/nemotron-3-super-120b-a12b` | Pinned IDs, not `auto/*`. |
| `VISION_MODEL` | `nvidia/meta/llama-3.2-11b-vision-instruct` | **No consumers** — the vision path is a standalone probe, not a feature. |
| `EMBEDDINGS_PROVIDER` | `stub` | `openai` / `stub` / `ollama`. There is no `sentence_transformers` option. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `ollama/nomic-embed-text` / `768` | The `ollama/` prefix is stripped before the request. |
| `RETRIEVAL_TOP_K` | `10` | Dense stage asks for `2×`, candidate pool is `max(2k, 3·rerank_k)` = 30. |
| `RERANKER_TOP_K` | `5` | **This truncates the API to 5 chunks** — a k=10 result is not expressible until it is raised. |
| `RERANKER_PROVIDER` | `stub` | `stub` / `cohere` / `cross_encoder` |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_BM25_WEIGHT` | `0.6` / `0.4` | The `+0.25·coverage` term is a literal in `hybrid.py`, not configurable. |
| `RETRIEVAL_PAGE_QUOTA` | `0` (off) | `1` = best chunk per (document, physical page); `2` = conservative. The measured win. |
| `QDRANT_MODE` | `local` | `local` (in-process, single-writer lock) or `remote`. |
| `MONGODB_URL` | `mongodb://127.0.0.1:27019` | `MONGODB_TEST_URL` points at `:27018`; `REQUIRE_MONGO=1` by default. |
| `DATABASE_URL` | `sqlite:///./data/app.db` | Identities/workspaces only, since documents moved to Mongo. |
| `JOBS_MODE` | `inproc` | `inproc` or `celery` |

> `.env.example` also lists `VECTOR_STORE=atlas`, `VECTOR_DIM`, `EMBEDDINGS_SEARCH_INDEX_NAME`
> and `ATLAS_SEARCH_INDEX_NAME`. **`Settings` declares none of them** and ignores unknown
> keys, so setting `VECTOR_STORE=atlas` changes nothing — the vector store is Qdrant.

## Tests

```powershell
docker compose -f infra/docker-compose.yml up -d mongo-test   # required, see below
cd backend
..\.venv\Scripts\python.exe -m pytest
```

1017 tests are collected across 25 files; a plain run on 2026-10-19 with
`finrag-mongo-test` up gives **1006 passed, 11 xfailed, 0 skipped in 57 s**. Unit
and HTTP tests run with deterministic stub providers — no API keys, no spend, no
network — and cover query understanding, chunking, page provenance, table
extraction, glossary mapping, citation verification, eval metrics, the financial
tools, tenant isolation, page-diversity selection, and a full HTTP smoke test.

Two things make a green run meaningful rather than decorative:

- The Mongo suites are parameterised `mock` + `real`, and `REQUIRE_MONGO` defaults to
  **on**: an unreachable `:27018` is reported as errors, not skips. Opt out with
  `REQUIRE_MONGO=0` only when you mean to.
- Isolation was **mutation-tested**: deleting the tenant filter from
  `BaseRepository._scoped` fails 320 tests across the three Mongo suites (146 from the
  416-case matrix alone).

Per `CLAUDE.md`, any number you quote must come from a plain `python -m pytest -q` in
`backend` — no `--deselect`, `-k`, or `-x`. A recorded offline run reads
`685 passed, 326 deselected, 6 xfailed`, which is a subset and must not be described as
the suite passing.

## Repository layout

```
investment-research/
├── backend/
│   ├── app/
│   │   ├── core/         # config, security, logging, errors
│   │   ├── providers/    # llm, embeddings, reranker, vectordb
│   │   ├── ingestion/    # parser, ocr, cleaner, tables, chunking, pipeline
│   │   ├── retrieval/    # query_understanding, hybrid, parent expansion
│   │   ├── reasoning/    # orchestrator, agents, tools (SQL), reports
│   │   ├── evaluation/   # offline harness + metrics
│   │   ├── api/          # FastAPI routers
│   │   ├── db/           # SQLAlchemy models
│   │   ├── jobs/         # in-proc worker (Celery-shaped)
│   │   ├── seed/         # sample docs + gold Q&A
│   │   ├── citation.py   # NLI verification
│   │   ├── conversation.py
│   │   ├── cli.py
│   │   └── main.py
│   └── tests/
├── frontend/             # Next.js 14 + React + TS
│   └── src/{app,components,lib}
├── infra/                # docker-compose.yml — the default way to run
├── docs/                 # design spec
├── start.bat             # docker compose up --profile app + health wait
├── stop.bat              # stops the web + API containers
└── prd.md
```

## Security notes

- The `.env` file is git-ignored. **Never commit API keys.** If a key is accidentally shared, rotate it immediately in the provider dashboard.
- Isolation is enforced at the repository layer: `BaseRepository._scoped` injects the caller's `tenant_id` into every Mongo query, a frozen `TenantContext` is required (missing context → 403), and the rule is mutation-tested rather than assumed.
- **Known open holes**, each pinned as a strict xfail in `tests/test_api_isolation.py` so the suite goes red the moment one is fixed:
  | Route | Problem |
  | --- | --- |
  | `POST /messages/{id}/feedback` | loads the message by id with no workspace or tenant check — tenant A can rate tenant B's answer |
  | `GET /admin/eval/runs` | no tenant filter, and the SQL `eval_runs` table has no tenant column to filter on |
  | `POST /auth/tenants` | unauthenticated: anyone can create tenants |
  | `POST /auth/users` | trusts a caller-supplied `tenant_id` while that tenant has no users, so an anonymous request can make itself admin of an unclaimed tenant |
  | `GET /auth/me`, `UserOut` | API keys are stored and echoed **in plaintext** |
- **Vector-level isolation is weaker than document-level.** Qdrant payloads carry `workspace_id` but not `tenant_id`, so a vector hit is only as well-scoped as the Mongo filter that follows it.
- Eval suites are addressed **by name inside `EVAL_SUITES_DIR` only**. The old `suite_path` field let an authenticated caller make the server open any file on disk; it was removed, not sanitised.
- MongoDB is unauthenticated on this box (`DEV_DISABLE_AUTH`), which is a dev-only posture. Localhost-bound ports are not a security control.
- The default SQLite + local Qdrant stack is fine for dev/demo. For production, use Postgres (or an authenticated Atlas deployment) + remote Qdrant + object storage.

## Status

Honest status, per [`docs/STATUS.md`](./docs/STATUS.md) and the measured artifacts under
[`reports/`](./reports). "Built" and "verified working end to end" are different claims and
this project has conflated them before, so they are kept apart here.

- **Phase 1 (MVP)** — built and verified: layout-aware parsing, hybrid retrieval, metadata
  filtering, parent-child chunking, page-accurate citations.
- **Phase 2 (Production)** — **partly**: structured financial store, tool-calling,
  conversation memory, document lifecycle and the evaluation framework all exist in code;
  the reranker is a stub, and on the real corpus the structured store holds **0 rows** from
  18 parsed tables.
- **Phase 3 (Enterprise)** — **partly**: orchestration, claim-level verification with
  fail-closed abstention, RBAC and multi-tenancy are implemented; five security holes above
  are open, and feedback logging is not audit-complete.
- **End-to-end answer quality is NOT verified.** On the 10-question PIF gold set with a real
  gateway LLM: 10/10 requests succeeded, but 0/8 numeric checks matched, 0/10 citations
  pointed at the right page, 2/10 answers were supported by their own cited evidence, and
  **0/10 passed all criteria**. Retrieval-side, post-rerank page diversity moves source+page
  recall 5/10 → 7/10 at k=10 — measured through the real `retrieve()` path, not a sandbox
  (`reports/pif_page_quota_1792426483019287700.json`). 7/10 is the ceiling for re-selection
  policies, as a **count** rather than a fixed set of questions: only two questions' evidence
  pages are in no variant's top 10, and a third is cutoff-limited rather than absent. The flag
  (`RETRIEVAL_PAGE_QUOTA`) ships **disabled by default**, and part of its gain is compensation
  for the stub reranker, so a real cross-encoder could shrink it.
- Vision/chart understanding is a **standalone probe, not a feature** (`VISION_MODEL` has no
  consumers; the probe read a bar chart to within 3.2% but only to the nearest gridline).
  Real-time market data, the calculation engine (`app/calc`), and FinQA/TAT-QA corpora do not
  exist.

A full write-up with the comparative tables and the artifact-to-claim map is
[`reports/finrag_academic_report.md`](./reports/finrag_academic_report.md).

## Operational gotchas

- **A Docker run and a host run do not share a database.** The containers keep
  identities in the `backend_data` volume (`/app/data/app.db`) and documents in
  MongoDB on `27019`; a host run uses `backend/data/app.db` and whatever
  `MONGODB_URL` points at (native mongod on `27017`, which has no Atlas Search).
  So seed, ingest and log in **inside one world** — a key from the host will get
  `401` from the container API, and vice versa.
- **The container API is published on 8010, not 8000**, because a host-side
  `ira serve` owns 8000. The web container still talks to `backend:8000` over
  the compose network.
- **Qdrant local mode is single-process.** Ingest documents *through the
  running server* (via the UI or the `/workspaces/{id}/documents` HTTP
  endpoint). If you ingest from a separate CLI process while the server
  runs, the server can't read those vectors and retrieval will return zero
  hits. For multi-process use, switch to `QDRANT_MODE=remote` and run a
  Qdrant server (see `infra/docker-compose.yml`).
- **API key precedence.** A `.env` file in the repo root wins over any
  globally-exported `OPENAI_API_KEY` shell variable, so a stale global key
  can't accidentally shadow your real one. In `APP_ENV=test` (set by the
  test suite) `.env` is never loaded — tests stay deterministic.
- **Stub vs live.** With no `OPENAI_API_KEY`, the LLM/embeddings providers
  fall back to deterministic stubs (offline, zero spend). All retrieval,
  citation, conversation, and evaluation logic runs identically; only the
  generated prose differs.
- **A `.env` edit does not reach a running backend.** `get_settings()` is
  `@lru_cache`'d, so the value is read once at import; and Compose sees no diff
  on an unchanged host path, so `up -d` alone is a no-op. You need
  `docker compose … restart backend` for a `.env` change and
  `up -d --build` for a code change (only `../.env` is bind-mounted read-only at
  `/app/.env`).
- **Some published PDFs carry doubled characters in their own glyph layer.**
  108 of 1,066 stored chunks (10.1%) over 61 of 157 pages of the PIF corpus hold
  text like `AAbboouutt PPIIFF`. Re-extracting all 157 pages fresh reproduces it
  on exactly the same pages and no page is "stored corrupted, PDF clean", so it
  is the source file, not our parser. Any fix needs a full reindex, which
  invalidates every published retrieval number — see
  [`reports/pif_doubled_text_evidence_1792430653471494700.md`](./reports/pif_doubled_text_evidence_1792430653471494700.md).

## License

MIT.
