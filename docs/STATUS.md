# FinRAG 2.0 — Implementation Status

Last verified: 2026-10-17. Every claim below was checked against the running
code and services on this machine, not against a plan.

## Verification artefacts

| Artefact | What it is |
|---|---|
| `docs/gateway-liveness.json` | Chat probe of every `nvidia/` model the gateway advertises |
| `docs/gateway-deep-probe.json` | Per-model chat/tools/json/stream/vision matrix for the survivors |
| `docs/gateway-embed-rerank.json` | Embedding + rerank probe (all failed) |
| `docs/gateway-stream-recheck.json` | Streaming recheck after the `reasoning_content` fix |
| `backend/scripts/verify_gateway.py` | Produces the above. Nothing mocked. |
| `backend/scripts/spike_atlas_search.py` | Atlas Search / Vector Search feasibility spike, 12 checks |

## Test suite

`305 passed, 0 skipped` from `backend/` with `.venv/Scripts/python.exe -m pytest`.

The MongoDB suite is **parameterised `mock` + `real`**. The `real` variants
**silently skip** unless a MongoDB answers on `mongodb://127.0.0.1:27018`
(`MONGODB_TEST_URL`). Without a server the suite reports green while 82 tests
never run. A server is required for a truthful run:

```powershell
docker run -d --name finrag-atlas -p 27018:27017 -e DEV_DISABLE_AUTH=1 mongodb/mongodb-atlas-local:latest
```

## Repository state vs. the brief

These corrections matter before any further planning:

- **`FINRAG_GAP_ANALYSIS.md` does not exist** in the repo root.
- **Not a git repository** — no `.git`, so "small commits" and "commit hash
  beside every benchmark number" are currently impossible.
- A prior session already built much of Step 1 and part of Step 2 (see
  `memory/mongodb-repository-layer.md`).
- **`backend/pyproject.toml` is not installable**: `readme = "../README.md"`
  points outside the package, so `pip install -e backend[dev]` fails.
  Dependencies must currently be installed by hand.

## Step 1 — Foundations: DONE

Structured JSON logging (`app/core/logging_structured/json_formatter.py`),
request/trace IDs (`app/core/tracing/trace_context.py`), error categories
(`app/core/errors.py`), config validation (`app/core/config.py`).

Note: `config.py` uses the aliases `OMNIROUTE_BASE_URL` / `OMNIROUTE_API_KEY`,
not the `LLM_BASE_URL` / `LLM_API_KEY` named in the brief. Model roles exist
(`chat_model`, `reasoning_model`, `vision_model`, `embedding_model`,
`rerank_model`, `fallback_chain`) but the **defaults hard-code model IDs**
(`nemotron-3-super-120b-a12b`, `nemotron-3-ultra-550b-a55b`) — the brief
forbids this. `nemotron-3-ultra-550b-a55b` is additionally *dead* on this
gateway.

## Step 2 — MongoDB layer: PARTIAL, NOT INTEGRATED

Built and tested: `app/db/mongodb/` (client, collections, tenant_context,
base_repository with enforced `tenant_id` injection, repositories, decimal
codec, bootstrap, pipeline guard).

Gaps against the brief:

1. **Zero app consumers.** Every router still imports SQLAlchemy
   (`api/auth.py`, `admin.py`, `chat.py`, `documents.py`, `deps.py`,
   `reports_jobs.py`); retrieval uses SQLAlchemy too. No production code
   constructs a Mongo repository or calls `ensure_collections`.
2. **Isolation coverage is incomplete.** 15 repository classes exist;
   isolation tests cover 6. Untested: `Tenant`, `Conversation`, `Job`,
   `EvalRun`/`EvalItem`, `AuditLog`, `Feedback`. **No HTTP route-level
   isolation tests exist at all**, and **Qdrant payload tenant scoping is
   untested** — the brief requires both.
3. **`Decimal128` is not enabled where money lives.** `decimal_codec` is wired
   into `base_repository`, but activation is per-class `decimal_fields`, which
   defaults to empty and **is declared by no production repository**. Money is
   still float at rest.
4. ~~Docker Compose has no `mongo` service~~ — resolved in (a) below.

## Round progress (Step 2 only)

### (a) Compose + containerisation — DONE, verified

`infra/docker-compose.yml` now provisions `mongo` (mongodb-atlas-local, host
27019), `mongo-test` (same image, host 27018, volume-free so tests can never
touch the app database), `qdrant`, an optional `redis` behind the `redis`
profile, and `backend`/`frontend` behind the `app` profile. Postgres is removed.
All published ports are host-overrideable (`MONGO_HOST_PORT`,
`BACKEND_HOST_PORT`, …) and bound to 127.0.0.1 only.

Three real problems found and fixed while verifying, not writing:

- **Port 27017 is not ours.** A native `mongod.exe` Windows service owns it and
  holds live data for other projects (`ai_coding_assistant`, `jurisai`,
  `ray_of_hope`, `FoodDelivery`, `food_ordering_db`) plus an older `finrag`
  database (collections: `researchsessions`, `documents`, `tenants`). Nothing
  there was touched. Our store moved to 27019, and `config.py`'s default
  `mongodb_url` follows, because a community mongod would connect fine and then
  fail every `$vectorSearch`/`$search` call. **Whether that old `finrag`
  database should be migrated is an open question for the user.**
- **Compose's `env_file` keeps inline comments** as part of the value, so
  `VISION_MODEL=  # e.g., nvidia/...` in `.env` was passed verbatim and the
  container refused to boot on config validation. The backend now mounts
  `.env` read-only at `/app/.env` and lets the app's own loader (which strips
  them correctly) parse it. `.env.example` documents the rule and uses
  own-line comments only.
- **Next.js compiles `rewrites()` into the routes manifest at build time**, so
  the runtime `BACKEND_URL` env var was ignored and `/api/health` proxied to
  `127.0.0.1:8000` (refused). Fixed with a build `ARG`.

Verified running: `finrag-mongo`, `finrag-mongo-test`, `finrag-qdrant`,
`finrag-backend`, `finrag-frontend` all healthy; `GET :8010/health` → 200;
`GET :3000/` → 200; `GET :3000/api/health` → 200 through the proxy. Backend and
frontend images build clean (835 MB / Node 20 multi-stage).

### (b) Silent skips — DONE, verified

An unreachable test MongoDB no longer reports as "skipped". `REQUIRE_MONGO`
defaults to **on**, and `mongo_support.enforce_reachable()` turns the condition
into a failure at all three former skip sites (`mongo_support._open_real`,
`test_mongo_hardening._open_real`, `test_pipeline_guard_integration.db`).
Opt out with `REQUIRE_MONGO=0` or `--allow-mongo-skip`.

Verified against a deliberately dead URL:

```
default policy        -> 22 passed, 21 errors   (loud)
REQUIRE_MONGO=0       -> 22 passed, 21 skipped
--allow-mongo-skip    -> 22 passed, 21 skipped
```

Documented test command (PowerShell), which is now the only honest one:

```powershell
docker compose -f infra/docker-compose.yml up -d mongo-test
cd backend
..\.venv\Scripts\python.exe -m pytest    # REQUIRE_MONGO is on by default
```


## Steps 3–12: NOT STARTED

| Item | State |
|---|---|
| `openai_compatible` provider | Absent. `config.py` accepts `llm_provider="omniroute"` but `get_llm()` has no branch for it, so it **silently returns `StubLLM`**. `OpenAIChatLLM`/`AnthropicLLM`/`OllamaLLM` read `settings` fields that no longer exist, so they would raise `AttributeError`. |
| Ollama zero-tool-call bug | Confirmed — `llm.py:465` returns `tool_calls=[]` unconditionally; `tools` is never sent. Same in `AnthropicLLM`. |
| Calculation engine (`app/calc`) | Absent. No provenance object anywhere. |
| Query route enum | Absent. Ad-hoc `mode: str` + `is_metric_lookup` booleans. |
| Silent filter-drop | Confirmed — `hybrid.py:185-188` drops the company filter on zero results; `query_understanding.py:86-93` drops the year filter; `hybrid.py:358` falls back to BM25-only on vector error. None abstain. |
| Persistent BM25 | Absent — `BM25Okapi` rebuilt over the candidate set every query (`hybrid.py:368`). |
| Verifier | `citation.py` uses token-overlap + 3-char number-prefix string matching; no Decimal comparison. `harness.py:_num_close` uses float with 0.01 tolerance. No ABSTAIN path. |
| Evaluation metrics | Placeholders — `context_precision` and `context_recall` are the *same* `n_retrieved/5` formula. No Recall@K, Precision@K, MRR, NDCG, EM or F1 exist. |
| FinQA / TAT-QA | Absent. |
| Market data | Absent. |
| Vision / chart module | Absent (`vision_model` config has no consumers). |
| Frontend login | Absent — `ConnectionGate.tsx` is a paste-API-key flow. |
| Auth holes | Tenant creation is unauthenticated (`auth.py:35`); API keys stored and echoed **in plaintext** (`models.py:74`, `security.py:98`); no login rate limiting. The prod `JWT_SECRET` guard is **already correct** (`config.py:246`). |

## Gateway reality (`http://localhost:20128/v1`)

Reachable and authenticating. **91 `nvidia/` model IDs are advertised, but only
13 answer.** 49 of the failures are upstream `404 not found`, 8 timed out at
60 s, plus 410 end-of-life (`nv-embedqa-e5-v5`) and transient 5xx.

Confirmed gateway behaviours relevant to benchmarking:

- The gateway **strips the provider prefix** in responses (`model` comes back as
  `nvidia/nemotron-3-super-120b-a12b`), so the returned model field cannot be
  trusted to equal the requested ID.
- It performs **per-model credential lockout with 2-minute cooldowns** and
  account fallback ("All credentials for model X are cooling down",
  `retry-after: 120`). A failed probe therefore poisons the next request to the
  same model, so benchmark runs must pin a single model and retry with backoff.
- `response_format: json_object` is **not honoured** by most models — they return
  prose. This makes the brief's schema-constrained JSON fallback for the
  SQL/calc tools mandatory, not optional.
- `tools` fails with `"auto" tool choice requires --enable-auto-tool-choice and
  --tool-call-parser` on several models (upstream vLLM launch config).

### Capability matrix (verified, non-specialised models)

| Model | chat | tools | json | stream | vision | chat ms |
|---|:-:|:-:|:-:|:-:|:-:|--:|
| `nvidia/meta/llama-3.2-11b-vision-instruct` | Y | Y | Y | Y | **Y** | 667 |
| `nvidia/poolside/laguna-xs-2.1` | Y | Y | Y | Y | – | 124 |
| `nvidia/nvidia/nemotron-3-super-120b-a12b` | Y | Y | Y | Y | – | 1594 |
| `nvidia/stepfun-ai/step-5-preview` | Y | Y | Y | Y | – | 1362 |
| `nvidia/z-ai/glm-5.3-flash` | Y | Y | n | Y | **Y** | 30040 |
| `nvidia/openai/gpt-oss-20b` | Y | Y | n | Y | – | 1432 |
| `nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b` | Y | Y | n | Y | untested¹ | 1350 |
| `nvidia/nvidia/nemotron-parse-2.0` | Y | n | Y | Y | – | 1743 |
| `nvidia/google/diffusiongemma-26b-a4b-it` | Y | Y | n | Y | – | 2469 |

¹ Catalogue declares no `input_modalities` despite being a VL model, so the
probe skipped vision. Needs a direct test before it is used for charts.

Specialised/safety models that answered chat but are not general-purpose:
`nemotron-3.5-content-safety`, `llama-3.1-nemotron-safety-guard-8b-v3`,
`riva-translate-4b-instruct-v2`, `muse-glimmer-30b`.

### Embeddings and reranking: NONE AVAILABLE

All 8 embedding candidates returned upstream `404`, and there is **no rerank
route**. Per the brief, embeddings must therefore come from a local model.
Neither option is currently installed on this machine: Ollama's binary exists
but the daemon is not running (`:11434` refused), and `sentence-transformers`
is not in the venv.

## Atlas Search spike: VIABLE

`mongodb/mongodb-atlas-local` runs correctly on this Windows/Docker box —
**12/12 checks passed**. `$vectorSearch` with a `tenant_id` filter returned only
the requesting tenant's chunks (7 ms warm), `$search` full-text with a keyword
tenant filter worked (102 ms), `Decimal128` round-tripped, replica-set
transactions committed, RSS ~810 MiB. Two gotchas found: exact-match filters
need `lucene.keyword` (not `lucene.keyword_analyzer`), and the score metadata
field is `$meta: "vectorSearchScore"`.
