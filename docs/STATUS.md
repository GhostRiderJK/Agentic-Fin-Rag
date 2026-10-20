# FinRAG 2.0 — Implementation Status

Last verified: 2026-10-19 (late round). Every claim below was checked against the
running code and services on this machine, not against a plan. Sections marked
*superseded* record what was true earlier and what replaced it, because the
measurements behind them are still cited elsewhere.

## Verification artefacts

| Artefact | What it is |
|---|---|
| `docs/gateway-liveness.json` | Chat probe of every `nvidia/` model the gateway advertises |
| `docs/gateway-deep-probe.json` | Per-model chat/tools/json/stream/vision matrix for the survivors |
| `docs/gateway-embed-rerank.json` | Embedding + rerank probe (all failed) |
| `docs/gateway-stream-recheck.json` | Streaming recheck after the `reasoning_content` fix |
| `backend/scripts/verify_gateway.py` | Produces the above. Nothing mocked. |
| `backend/scripts/spike_atlas_search.py` | Atlas Search / Vector Search feasibility spike, 12 checks |
| `reports/pif_page_quota_1792426483019287700.json` | Page-quota sweep through the production `retrieve()` path |
| `reports/pif_page_quota_broad_1792427509200136400.json` | Same, plus k=1..10 cutoff curves and the fan-out shape probe |
| `reports/pif_stage_final.json` | Offline stage-attribution ablation (11 configurations) — **see the staleness warning in the retrieval section before quoting it** |
| `reports/pif_prefusion_misses_1792427755170889700.json` | Candidate-generation cause for each gold question that misses k=10 |
| `reports/pif_doubled_text_1792430653471494700.json`, `reports/pif_doubled_text_evidence_1792430653471494700.md` | Doubled-character ingestion defect: provenance, counts, limitations |
| `docs/page-quota-serving-ab-protocol.md` | A/B protocol for the serving API — **written, not executed** |
| `backend/scripts/exp_pif_stages.py`, `exp_page_quota_broad.py`, `diag_pif_prefusion_misses.py`, `diag_pif_doubled_text.py` | Produce the four reports above; all re-runnable |

## Test suite

> Current baseline, re-run 2026-10-19 with `finrag-mongo-test` up:
> **1017 tests collected → 1006 passed, 11 xfailed, 0 skipped** in 57 s
> (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/Scripts/python.exe -m pytest -q`
> from `backend/`, no `--ignore`, no `-k`, no `--deselect`).
> The 11 xfails are 6 distinct route holes, mostly mirrored across the `mock`
> and `real` Mongo parameterisations, each pinned to the item that closes it:
> message feedback, `GET /admin/eval/runs`, `POST /auth/tenants`, anonymous
> tenant claim, and two API-key-echo tests. See *(e)* below.

The offline slice used while iterating (integration tests deselected, so **not**
a reportable number) is `685 passed, 326 deselected, 6 xfailed in 13.24s` —
`reports/offline_suite_after_eval_instrumentation.txt`.

Historical first-round baseline, kept for context only:
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
- ~~**Not a git repository**~~ — **resolved.** `.git` exists, the branch is
  `master`, and work is committed per change (`git log` shows the bcrypt fix,
  the Mongo event-loop bridge, the retrieval query layer, the omniroute
  provider and the page-provenance fix as separate commits). Commit hashes can
  now sit beside benchmark numbers, as the brief requires.
- A prior session already built much of Step 1 and part of Step 2 (see
  `memory/mongodb-repository-layer.md`).
- **`backend/pyproject.toml`** — the `readme = "../README.md"` field that
  pointed outside the package and broke `pip install -e backend[dev]` is
  **gone**; the file now declares `sqlalchemy`/`alembic` as the legacy store
  with a comment naming this document as the migration record. Editable
  install has not been re-run since the field was removed, so "installable" is
  still an unverified claim rather than a closed one.

## Step 1 — Foundations: DONE

Structured JSON logging (`app/core/logging_structured/json_formatter.py`),
request/trace IDs (`app/core/tracing/trace_context.py`), error categories
(`app/core/errors.py`), config validation (`app/core/config.py`).

Note: `config.py` uses the aliases `OMNIROUTE_BASE_URL` / `OMNIROUTE_API_KEY`,
not the `LLM_BASE_URL` / `LLM_API_KEY` named in the brief (both names work —
`AliasChoices` maps them, and the brief's names win). Model roles exist
(`chat_model`, `reasoning_model`, `vision_model`, `embedding_model`,
`rerank_model`, `fallback_chain`) but the **defaults hard-code model IDs**
— the brief forbids this. Against `docs/gateway-liveness.json` (74 probed, **13
that answer chat**), the pinned `chat_model`/`reasoning_model`
(`nemotron-3-super-120b-a12b`) and the first three `FALLBACK_CHAIN` entries are
all live; the dead `nemotron-3-ultra-550b-a55b` (upstream 404) is no longer a
default. One real gap remains: the **fourth** fallback entry is
`ollama/llama3.1`, and `ollama list` shows only `nomic-embed-text` and
`llama3.2:3b`, so the tail of the chain cannot actually be used on this machine.

## Step 2 — MongoDB layer: PARTIAL, **now with real app consumers**

Built and tested: `app/db/mongodb/` (client, collections, tenant_context,
base_repository with enforced `tenant_id` injection, repositories, decimal
codec, bootstrap, pipeline guard, and — since commit 3dd6484 — `runtime.py`,
one background event loop that lets synchronous FastAPI handlers call the async
repository layer through `run_sync`).

Gap 1 is **closed** (it read "Zero app consumers"). Production code now goes
through Mongo:

| Path | Store today |
|---|---|
| `api/documents.py` — document list/get/patch/delete, chunk reads, structured-financial reads | **Mongo** (`DocumentRepository`, `ChunkRepository`, `StructuredFinancialRepository`) |
| `ingestion/pipeline.py`, `ingestion/reindex.py` | **Mongo** |
| `retrieval/hybrid.py` — the candidate corpus and parent lookup | **Mongo** |
| `api/chat.py`, `api/deps.py` | construct a `TenantContext` for the paths above |
| tenants, users, workspaces, conversations, messages, jobs, reports, eval runs | **still SQLAlchemy** (`app/db/base.get_db` in `auth.py`, `admin.py`, `chat.py`, `documents.py`, `reports_jobs.py`, `deps.py`) |

So the system is **dual-store**: document evidence in MongoDB, identity and
conversation state in SQLite. That is a real state, not a bug, but it means the
tenant-scoping guarantees of `BaseRepository._scoped` do **not** yet apply to
the routes where the leaks were found.

Live confirmation of exactly how far the migration got (`finrag` on 27019, read
2026-10-19): **3 collections exist — `documents` (5), `chunks` (2136),
`structured_financials` (0)** — not the 13 the layer defines. Ten of the
collections have never been written by the running app, which is the same fact as
the table above stated from the other side. And `structured_financials: 0` is
still true with the Mongo writer in place, so the zero is an extraction/persist
problem, not a legacy-SQL artifact.

Remaining gaps against the brief:

1. ~~Zero app consumers~~ — closed, table above.
2. **Isolation coverage is complete at the repository layer** (item (c) — the
   416-case matrix over all 12 tenant-owned repositories, mutation-tested), and
   route-level coverage exists (item (e)). What is still open is the **6 unmet
   routes** pinned as xfail, all of them on the un-migrated SQL side.
   **Qdrant payload tenant scoping is still untested** — chunk payloads carry no
   `tenant_id`; the workspace filter is the only boundary.
3. **`Decimal128` does govern the money path that has moved.** Ingestion
   persists extracted table financials through `StructuredFinancialRepository`
   (`pipeline.py:44`, `_persist_financials` at `:209`), so those rows go through
   the codec. What has *not* moved is the job row: `app/jobs/queue.py` still
   creates SQLAlchemy `Job` objects (`:42`, `:62`, `:84`), so
   `JobRepository.decimal_fields = {"progress"}` currently protects a collection
   nothing writes to.
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


### (c) Isolation coverage for every repository — DONE, verified

**Count correction.** The brief said "all 15 repositories". `app/db/mongodb`
has **13 repository classes over 13 collections**: 12 tenant-owned
(`collections.TENANT_OWNED`) plus `tenants`, which is the tenancy root and
carries no `tenant_id`. `FINRAG_GAP_ANALYSIS.md` does not exist in the repo, so
the 15 cannot be reconciled against it; coverage below is complete for what is
actually in the tree.

New `tests/test_mongo_isolation_matrix.py` (416 tests) applies one battery
mechanically to all 12 tenant-owned repositories, including the domain methods
each subclass adds — `get_by_email`, `get_by_api_key`, `list_for_tenant`,
`list_for_workspace`, `find_by_metadata`, `set_status`, `list_for_document`,
`list_children`, `get_many_by_ids`, `get_metric`, `compare_metric`,
`list_metrics`, `list_for_conversation`, `set_feedback`, `list_recent`,
`list_for_run`, `record`, `list_for_message`:

- read: foreign id invisible via `get_by_id`/`exists`/`find_one`; unfiltered
  `find_many`/`count`/`stream` scoped; a caller-supplied `tenant_id` (including
  `{"$ne": own}`) cannot widen scope; `$or` over both ids returns only own;
  projection reads scoped; aggregation cannot span tenants.
- write: foreign id cannot be updated or deleted; `update_many`/`delete_many`
  touch only own rows; inserts cannot claim another tenant; `insert_many`
  stamps every row; ownership cannot be transferred by `$set`; domain mutation
  probes must return 0 for foreign ids.
- context: every operation raises `TenantContextError` without a context, and
  `stream()` is rejected before iteration rather than at first `await`.

Three meta-tests guard against drift, which matters more than the individual
cases: a new collection in `TENANT_OWNED` without a matrix case fails, a new
`BaseRepository` subclass pointed outside `TENANT_OWNED` fails, and a matrix
case wired to the wrong repository class fails.

**The suite was mutation-tested, not just run.** Deleting the tenant filter
from `BaseRepository._scoped` produced **320 failures** across the three Mongo
suites (146 from the matrix alone), so the 416 passes are real signal, not
assertions that cannot fail. The mutation was reverted with
`git checkout -- backend/app/db/mongodb/base_repository.py`.

Two defects fixed while getting there:
- `test_mongo_isolation.py` had an assertion merged onto the same line as a `#`
  comment, so it never executed; restored to a real assert.
- `test_mongo_hardening.py` carried verbatim copies of `mongo_support`'s
  `_PROBES`, `_unreachable_reason`, `run_db` and `_open_real`, shadowing its own
  imports. Those and its `run`/`db`/`real_db`/`ctx_*` fixtures now live only in
  `mongo_support.py` and are imported, so the mock/real policy and
  `REQUIRE_MONGO` enforcement have exactly one definition.

```
mongo suites (isolation + matrix + hardening)  ->  541 passed
whole backend suite, mongo-test up             ->  733 passed, 0 skipped
```

### (d) Decimal128 on every money/ratio repository — DONE, verified

The codec in `app/db/mongodb/decimal_codec.py` only enforces what a repository
*names* in `decimal_fields`; an undeclared column is stored as a float and no
error is raised anywhere. So the declarations are the feature, and they are now
on the two repositories that hold money or ratios:

- `StructuredFinancialRepository.decimal_fields = {"value", "confidence"}` —
  `value` is money or a ratio depending on the metric, `confidence` is a 0..1
  ratio.
- `JobRepository.decimal_fields = {"progress"}` — a 0..1 ratio that is compared
  and reported to users.

That is the complete set. Every other numeric column in the legacy SQL schema
is `Integer`/`BigInteger` (`token_count`, `page_count`, counters) and correctly
stays a Python `int`; nothing there was converted.

New `tests/test_decimal_fields.py` (13 tests, 23 cases once the mock/real
pairing is applied) proves the behaviour rather than the intent:

- round trip keeps the *scale*, not just the value — `Decimal("1234.5600")`
  reads back with `str(...) == "1234.5600"`, which silent normalisation would
  break, and the stored BSON type is `Decimal128` (read bypassing the repo).
- a `float` is refused on insert (`value` and `confidence`), on filter
  (`{"value": 1234.56}`) and on `$set` update, and the refused insert leaves
  the collection empty — no partial write.
- `bool` and `NaN` are refused; a value with more than 34 significant digits is
  refused rather than rounded.
- `{"$sum": "$value"}` over `0.1` and `0.2` is exactly `0.3` in aggregation.
- `get_metric` returns `Decimal`, not `float`.
- **drift guard**: every `sa.Float` column in `app/db/models.py` must appear in
  some repository's `decimal_fields` under the same `collection.field` name.
  Adding a float money column without deciding its Mongo counterpart fails
  here. It retires with the SQLAlchemy layer at item (i).

Mutation-checked: setting both `decimal_fields` to `frozenset()` produced
**15 failures** in this file, so the passes are signal.

The matrix and `test_mongo_isolation.py` financial/job payloads were switched to
`Decimal` accordingly — they had been seeding floats, which the codec would have
rejected the moment the declarations landed.

```
mongo suites (isolation + matrix + hardening + decimal)  ->  564 passed
whole backend suite, REQUIRE_MONGO=1, mongo-test up      ->  756 passed, 0 skipped
```

Write coverage at the time of *(d)*: no route wrote through these two
repositories yet, so the declarations could not break a live path. That has
since changed for money — `app/ingestion/pipeline.py` and
`app/seed/seed_financials.py` now persist extracted financials through
`StructuredFinancialRepository`, which is why the seeding payloads in
`tests/test_api_isolation.py` and the matrix were switched to `Decimal` first.
`JobRepository` is still read-only in practice; jobs remain SQL.

### (e) Route-level isolation tests — DONE, 8 holes found and pinned

New `tests/test_api_isolation.py` (30 tests) drives the real FastAPI app with
two tenants, each with its own admin, workspace, document, conversation,
message, job and eval run, and calls every id-taking route with the *other*
tenant's credentials. Foreign rows are re-read straight from the database, so a
"404" that quietly mutated something still gets caught.

`COVERAGE` maps every tenant-facing route in the OpenAPI document to the case
that checks it, and `test_every_route_is_accounted_for` fails if a route is
added without one — or if a mapped route disappears. Mutation-checked: deleting
the `GET /jobs` row makes that meta-test fail.

**Fails closed (22 tests):** workspace-scoped document list/get/patch/delete,
document upload rejected before any file or row is written, `/chat` and
`/chat/stream` on a foreign workspace, conversation list and message history,
`/reports` on a foreign workspace, `/jobs` list (including a `workspace_id`
filter that cannot widen scope) and `/jobs/{id}`, eval on a foreign workspace,
cross-tenant user creation (403), all 15 data routes answering 401 without
credentials, and unknown keys rejected.

**Leaks (8 tests, `xfail(strict=True)` — the assertion is the requirement, the
marker records that the route does not meet it, and strictness turns the suite
red the moment it is fixed):**

| Route | Leak | Closes at |
|---|---|---|
| `POST /messages/{id}/feedback` | `app/api/chat.py:180` loads the message by id with no workspace or tenant check; tenant A rates tenant B's answer and the row changes (observed 200) | (f) chat migration |
| `GET /admin/eval/runs` | `app/api/admin.py:65` has no tenant filter — and `eval_runs` has **no tenant column** in the SQL schema, so it cannot be filtered there at all | (f) |
| `POST /auth/tenants` | no authentication at all (`app/api/auth.py:36`): anyone can create tenants | (h) |
| `POST /auth/users` | trusts the caller-supplied `tenant_id` whenever that tenant has no users yet (`app/api/auth.py:58`), so an anonymous request can make itself admin of any unclaimed tenant | (h) |
| `GET /auth/me`, `UserOut` | the API key is stored in plaintext and handed back by every user response | (h) |
| password auth | **not a leak, a break:** passlib 1.7.4 cannot run its bcrypt backend detection against the installed bcrypt 5.0.0 (its probe secret exceeds bcrypt's 72-byte limit), so `hash_password()` raises `ValueError` on first use — creating a user with a password dies (a 500 in a running server) and `/auth/login` can therefore never succeed. API-key auth is the only working path, which is why this stayed invisible | **fixed**, see below |

Two further observations that are not route leaks but are recorded for (f):
`users.email` is globally unique in SQL while the Mongo index plan makes it
unique *per tenant*, so the login lookup must be re-scoped during migration; and
`test_smoke`'s "eval run" step in its docstring is never actually exercised
(superseded — the eval run is now exercised there, see the eval-job section).

```
tests/test_api_isolation.py                          ->  22 passed, 8 xfailed
whole backend suite, REQUIRE_MONGO=1, mongo-test up   ->  778 passed, 8 xfailed, 0 skipped
```

Incidental fix: the local Qdrant client's destructor was running during
interpreter shutdown and printing an `ImportError` that no test caused; the
session fixture in `tests/conftest.py` now closes it explicitly.

### Password hashing: passlib dropped, login works again — DONE, verified

Approved before starting: the auth router migration in (f) would otherwise have
ported a login path that cannot work. `app/core/security.py` now calls `bcrypt`
directly (same `$2b$` ident, same cost 12 passlib used, so existing hashes still
verify) and `passlib[bcrypt]` is replaced by `bcrypt>=4.0` in
`backend/pyproject.toml`.

Behaviour pinned by `tests/test_password_hashing.py` (10 tests):

- round trip, wrong password, salted-and-distinct, `$2b$12$` format/length.
- `password_hash=""` (a user created without a password) can never log in,
  including against an empty request.
- a malformed stored hash returns False instead of raising — passlib's blanket
  `except Exception` hid that class of bug rather than handling it.
- non-ascii secrets work.
- the 72-**byte** limit is inclusive and measured in bytes, not characters: a
  longer secret is refused with a 422 rather than silently folded to 72 bytes,
  which would have made different passphrases the same credential. On the login
  side an over-long secret simply cannot match.

The strict xfail on `test_a_user_can_be_created_with_a_password` then flipped to
a failure exactly as designed, and was replaced by
`test_a_user_can_be_created_with_a_password_and_logs_in`, which now also proves
the JWT it returns authenticates `/auth/me` and that a wrong password gets 401.

```
whole backend suite, REQUIRE_MONGO=1  ->  789 passed, 7 xfailed, 0 skipped
```

### Checkpoint decisions 2 and 4 — DONE, verified

**Decision 4 (models, config only).** `config.py` now defaults
`CHAT_MODEL`/`REASONING_MODEL` to `nvidia/nvidia/nemotron-3-super-120b-a12b`,
`VISION_MODEL` to `nvidia/meta/llama-3.2-11b-vision-instruct`, and
`FALLBACK_CHAIN` to the four pinned ids in order. The dead
`nemotron-3-ultra-550b-a55b` default is gone from both `config.py` and
`tests/conftest.py`. `LLM_BASE_URL`/`LLM_API_KEY` are accepted through
`AliasChoices` and win over the `OMNIROUTE_*` names, which still work.
`test_config.py` pins these values so a future default cannot drift back to a
dead model.

**Decision 2 (embeddings).** `OllamaEmbeddings` calls the native
`/api/embed` route; `EMBEDDING_MODEL` and `EMBEDDING_DIM` are config, and the
`ollama/` role prefix is stripped before the request. It fails loudly rather
than degrading: an unreachable daemon, a model that is not pulled, a short
answer, or a width that disagrees with `EMBEDDING_DIM` all raise `ProviderError`
with the exact command to run. `sentence-transformers` is removed from the
`local` extra and the provider option that used it is deleted; the reranker's
`cross_encoder` path still degrades to the stub with a warning.

Real run against the daemon (not mocked), 768-dim, L2-normalised:

```
provider= ollama dim= 768
dims= [768, 768, 768]   l2_norms= [1.0, 1.0, 1.0]
cos("revenue grew 12%…", "gross margin compressed…") = 0.4679
cold first call = 32.2 s (model load); tests use an httpx MockTransport
```

The daemon was not running. I started `ollama serve` and ran
`ollama pull nomic-embed-text` (274 MB) to verify this for real.

**Defect found while doing it.** `ProviderError.__init__` required `provider`
positionally, and all 21 raise sites across `app/providers/*` omit it — so
*every* provider failure surfaced as `TypeError` instead of a 502. `provider`
is now optional. Separately, `openai_api_key`/`anthropic_api_key` and the
`*_chat_model` fields were read by the providers but never declared on
`Settings`; they are declared now, as `SecretStr`, with the three call sites
updated to `get_secret_value()` (a bare `SecretStr("")` is truthy, so the old
"is the key set?" checks would have passed with no key).

### Chart-reading probe (decision 4, recorded)

`backend/scripts/probe_vision_chart.py` draws a bar chart with PIL from an
explicit ground truth (`Q1 1240, Q2 1585, Q3 1372, Q4 1908` USD m), sends it as
a PNG, and scores the reply. Evidence: `docs/vision-chart-probe.json`, image
sha256 `ec87a5389a689e08…`, seed 7, 2026-10-17.

| Model | Result | Latency | prompt/completion tokens |
|---|---|---|---|
| `nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b` | `{1200, 1600, 1400, 1900}` — max err 40 (3.23%) | 47.0 s | 632 / 1539 (1497 reasoning) |
| `nvidia/meta/llama-3.2-11b-vision-instruct` | `{1200, 1600, 1400, 1900}` — max err 40 (3.23%) | 16.2 s | 6477 / 34 |

Both models read only to the nearest gridline; neither resolves a value between
gridlines, and they agree exactly, so this is a capability ceiling rather than
one weak model. At 400 `max_tokens` nemotron-3.5-super-vl returned HTTP 200 with
**no answer at all** (400/400 tokens spent reasoning) — a config-trap worth
remembering for Step 4. Consequence for the design: chart images may inform
narrative, but every reported number must come from document text or tables,
never from a chart read. Note also the 10x prompt-token difference for the same
image.

## Eval runs are queued jobs, and suites are named — DONE, verified 2026-10-19

Two defects found while browser-verifying the UI against the real Kaggle corpus
(both measured, not inferred):

1. `POST /admin/eval/run` answered the request only after running the whole
   suite — 431.8 s of LLM calls. The Next.js `/api` proxy gives up at ~30 s, so
   the dashboard showed `socket hang up` / HTTP 500 while the run went on to
   finish server-side and record itself. Every chat longer than half a minute
   fails the same way; eval made it impossible to miss.
2. `api.runEval` sent no suite, so the button always ran the seeded ACME demo
   set — there was no way to evaluate a real corpus from the UI.

**Fix.** The route queues work and returns immediately; the UI polls.

| Change | Where |
|---|---|
| `enqueue_eval()` + `kind == "eval"` worker branch (`_do_eval`) | `app/jobs/queue.py` |
| `POST /admin/eval/run` → `202 {job_id, status, suite, workspace_id}` | `app/api/admin.py` |
| `GET /admin/eval/suites` — suites listed by name | `app/api/admin.py` |
| Suite allowlist: names resolve inside `EVAL_SUITES_DIR` only | `app/evaluation/suites.py`, `app/core/config.py` |
| Suite dropdown, 2 s job poll, live metrics from the job, suite column in history | `frontend/src/app/eval/page.tsx`, `frontend/src/lib/api.ts` |

A suite is now addressed by **name, never by path**. `EvalSuite.from_jsonl`
reads whatever path it is handed, so the old `suite_path` field let an
authenticated caller have the server open any file on disk; `resolve()` accepts
`[A-Za-z0-9][A-Za-z0-9._-]{0,99}` and appends `.jsonl` inside
`./data/eval_suites` (`/app/data/eval_suites` in the container, created at
startup by `ensure_dirs()`). `""`/`"default"`/`"packaged"` mean the seeded set.

**Side effect, not planned but real:** the route used to report the newest
`eval_runs` row with no tenant filter, which was on the leak list — tenant A
could be handed tenant B's run id. It returned no run row at all now, so that
xfail became a passing test (`test_eval_run_is_queued_and_names_no_run_row`).
`GET /admin/eval/runs` still has no tenant filter and stays xfailed for (f).

**Tests.** `tests/test_smoke.py` now exercises the docstring's "eval run" claim:
suites listed, run queued, job polled to `succeeded`, `n_items == 7`, metrics
present, run in history — against both `mock` and `real` Mongo. Its user is
created with `role: "admin"` because the eval routes are admin-only and the user
route honours the requested role. `tests/test_units.py` pins suite discovery and
rejection. `COVERAGE` gained the `GET /admin/eval/suites` row.

```
tests/test_api_isolation.py                ->  53 passed, 11 xfailed
tests/test_smoke.py                        ->  3 passed
whole backend suite, at that round          ->  834 passed, 11 xfailed, 0 failed
```

(11 xfailed already — the same 6 route holes as today; the suite total has since
gone 834 → 1006 as the retrieval, page-provenance and gold-attribution tests
landed. See *Test suite* at the top for the current number.)

**Live verification** (containers rebuilt, `pif_gold_qa.jsonl` + a 2-item
`pif_gold_short.jsonl` copied into `/app/data/eval_suites`, real OmniRoute LLM):

- `POST /api/admin/eval/run` through the Next proxy → **HTTP 202 in 0.035 s**;
  `GET /api/jobs/{id}` polled every 15 s for ~215 s, every response 200, then
  `succeeded` with the metrics in `job.result`. The run outlived the old 30 s
  cutoff by 7x without a single failed request.
- `/eval` in the browser: dropdown listed
  `packaged — 7 items (seeded) / pif_gold_qa — 10 items / pif_gold_short — 2 items`,
  the button read `Running…` while 18+ polls went out all 200, no error banner,
  and history showed a `Suite` column with the new `pif_gold_short` run.

Still open, unchanged by this work: naive-UTC timestamps render ~5.5 h off,
`structured_financials` is 0 despite 18 tables parsed, and the chat page's
suggested questions and report form defaults are hardcoded to ACME.

**Long chats — resolved 2026-10-19.** The 30 s abort was the Next.js `/api`
proxy's default; `frontend/next.config.js:11` now sets
`experimental.proxyTimeout: 900_000`, which is what let the 215 s eval poll
above finish. Chats are still **in-line** requests — `POST /chat/stream`
exists (`app/api/chat.py:89`) but `frontend/src/lib/api.ts` never calls it and
chat is not a queued job — so the *architecture* complaint stands: a restart
mid-chat loses the generation. They no longer die at half a minute.

## Page provenance: every chunk claimed page 1 — **fixed**, commit c6cae69, verified 2026-10-19

`ingestion/chunking.py:_split_sections` took the whole cleaned document as one
string and hard-coded `page=1`, discarding the page numbers the parser had
already resolved. On a real 80-page report every parent and child chunk claimed
the cover, so citations pointed at pages that did not contain them and no
page-level retrieval metric could be computed honestly.

Section detection now walks page by page: a header stays current across a page
break, but the body it emits is page-bounded, so a section spanning pages 12–14
becomes three chunks on those three pages. Pinned by the added cases in
`tests/test_units.py`.

Evidence the live store reflects the fix rather than the old hard-code: the PIF
runs in the next section report per-chunk `physical_page` values spread across the
document (53, 59, 64 …) with a mean of 3.6 unique pages over 8 returned chunks at
quota 0. A pre-fix store would report 1.

**This is what makes every page-diversity number below meaningful**, and it is
also why the doubled-character remediation is so expensive: fixing it the obvious
way means re-deriving chunk text and re-vectorising, which invalidates the page
inventory measured here.

## Retrieval page diversity: measured, implemented, **disabled by default** — verified 2026-10-19

The one strict win found in this system is not a model change but a **selection
rule**: after reranking, group the returned candidates by
`(document_id, physical_page)` and keep at most `RETRIEVAL_PAGE_QUOTA` chunks per
group (`_page_diversity_select`, `retrieval/hybrid.py:274`, applied at
`:497-498` in place of the `out[:rerank_top_k]` truncation). Default **0 = off**;
`.env` does not set the key, only `.env.example:98` documents it, so **the live
environment is unchanged** and turning it on needs a separate decision.

### Production-path numbers, not a sandbox

`reports/pif_page_quota_1792426483019287700.json` calls the real
`HybridRetriever.retrieve()` against the live stores
(`chunks_nomic768_test`, Mongo 27019, Ollama embeddings, stub reranker), 10 gold
PIF questions, `RETRIEVAL_TOP_K=10`, `RERANKER_TOP_K=10` applied by a private
`Settings.model_copy` — never by editing `.env`. `generation_calls: 0`,
`writes_to_datastore: 0`, `scoping_ok: true` for every arm.

| Quota | source+page recall @5 | @10 | mean unique pages @10 | expected page in the final 8 | in returned 8 |
|---|---|---|---|---|---|
| 0 (today) | 3/10 | 5/10 | 3.6 | 4/10 | 4/10 |
| **1** | **6/10** | **7/10** | **9.2** | **7/10** | **7/10** |
| 2 | 5/10 | 7/10 | 7.1 | 6/10 | 6/10 |

Quota 2 is dominated by quota 1 at six (question, cutoff) pairs, so **1 is the
recommended value**. Source-only recall is 10/10 at every cutoff in every arm —
the system always finds the right *document*; it was burying the right *page*.
These match the offline ablation's `baseline` and `post_div_page1` rows, which is
what makes the ablation trustworthy rather than decorative.

The workspace under test holds exactly 2 documents (`2024-en_2024-en.pdf`,
`2023-en_2023-en.pdf`); the broad report also records
`documents_outside_workspace: 3` — three more documents in the same MongoDB that
belong to *other* workspaces. Every arm is therefore checked to have returned
nothing outside the probe workspace (`scoping_ok: true`), which is the only
tenant boundary that exists on the retrieval path today.

### Honest limits on that result

- **n = 10.** 3 questions gained, 0 lost gives an exact McNemar (sign) two-sided
  p ≈ 0.25 at k=5 and ≈ 0.5 at k=10. Deterministic and reproducible, **not**
  statistically proven.
- **7/10 is the ceiling, as a count and not a fixed set.** Widening dense
  recovers pif-002 and loses pif-009 — net zero. Only **pif-007 and pif-010**
  (page 6) are in no variant's k=10 hit set; pif-002 is cutoff-limited, not
  absent (its page-64 rows sit at fused ranks 54/65/70/74/76/80 and the best
  reaches rank 19 on a 120-hit probe, i.e. inside the pool).
- **The gain partly compensates for the stub reranker.** With reranking switched
  off entirely (`rerank_passthrough`) mean unique pages @10 is already **7.5**
  and the quota on top of that reaches only 5/10 @10 (vs 7 for the stub). The
  stub concentrates pages; the quota undoes that concentration. A real
  cross-encoder could shrink the benefit — and no cross-encoder is installable
  here, so that has to stay a caveat rather than be measured.
- The un-golded fan-out paths change shape too: `_comparison_retrieve` /
  `_multi_year_retrieve` still return 8 chunks with both years represented, but
  unique pages go 5→7 and 7→8 (`fanout_probe`, explicitly *not* scored against
  gold answers).
- The saved reports were run against **gold v1** (`pif_gold_qa.jsonl`, sha256
  `00f240c9…154512ce`). v2 (`pif_gold_qa_v2.jsonl`, stable ids + explicit
  attribution) now exists and every consumer loads it through
  `pif_chat_check.load_gold()`; v2 attributes all ten questions to `2024-en.pdf`,
  dropping v1's extra `2023-en.pdf` for pif-008 — no metric changes, because that
  hit already came from `2024-en`.
- ⚠️ **`reports/pif_stage_final.json` (21:22) predates the `StageProbe` stage
  alias fix** (`inventory` / `corpus` / `fused_pool`, each with its own
  `stage_sizes`) and the `exp_pif_stages.py` edit at 22:31. Its *selection*
  metrics agree with the production run above, but any **per-question stage
  rank** taken from it must be re-generated before it is quoted in a submitted
  document.

### The doubled-character defect found while chasing pif-002

108 of 1,066 stored PIF chunk rows (**10.1%**) across **61 of 157 pages**
(`2023-en` 78 chunks / 41 pages, `2024-en` 30 / 20) hold text with every
character repeated — `about` → `AAbboouutt`. Evidence and method:
`reports/pif_doubled_text_1792430653471494700.json`,
`reports/pif_doubled_text_evidence_1792430653471494700.md`,
`backend/scripts/diag_pif_doubled_text.py`.

The decisive measurement is the cross-tabulation: re-extracting all 157 pages
fresh reproduces the defect on **exactly the same 61 pages**, and **0** pages are
"stored corrupted but PDF clean". The only cell that could have implicated our
parser or chunker is empty. OCR never ran on these files, and the child overlap
at `chunking.py:214-222` is word-boundary aligned so it cannot produce
intra-word doubling. So the defect is in the PDF's own glyph layer, and the
corpus shipped that way. pif-002's corrupted page-64 chunk is a contributing
factor in its miss, **not** its established cause — the two other
answer-carrying children on that page store clean text and still miss the pool.
pif-007/pif-010 have **zero** corrupted chunks on their expected page; their
cause is candidate generation (empty `section`, no dense hit inside `top_k*2`).

**Remediation is deliberately not applied.** `page.dedupe_chars()` /
position-dedup in `parser.py:100-102` takes every sampled corrupted page to 0
expanded tokens, but it requires a full PIF **reindex** (Mongo text + Qdrant
vectors), which invalidates **every retrieval number in this document**. Reindex
and the quota decision must not move together.

### The serving API cannot express the measured configuration

- `.env` sets `RERANKER_TOP_K=5`, which truncates the returned set to 5 chunks
  (pool `max(2*10, 3*5)` = 20), so k=10 is unmeasurable through `/chat` as
  configured: `pif_chat_check`'s `retrieval_at_k["10"]` reads
  `k_complete: false` for every question. **Arm A of any A/B therefore has to
  raise it to 10, which means the control is not today's behaviour.**
- The quota code is **not deployed**: the `backend` image is built from source,
  and only `../.env` is bind-mounted read-only at `/app/.env`
  (`infra/docker-compose.yml:92-110`). Changing the code needs `up -d --build`;
  changing `.env` needs `restart`, because `get_settings()` is `@lru_cache` and
  Compose sees no diff on an unchanged host path, so `up -d` alone is a no-op.
- `/chat` has no dry-run: it always writes a conversation plus 2 SQL message rows
  and always generates (tool loop `max_iterations=3` + at most one
  citation-format repair ⇒ 1–4 completions per call).

`docs/page-quota-serving-ab-protocol.md` is the written response to all of that —
quota 0 vs 1 at k=10 through `/chat`, `[APPROVAL]`-gated rebuild and `.env` edit,
read-only preflight (including an admin role check, because
`include_evidence_trace` is admin-only), deterministic retrieval-shape metrics as
the primary endpoint and noisy answer-level metrics as secondary, acceptance
criteria, a 40-request / 40–160-completion cost bound, and rollback. **It has not
been executed.** It recommends approving a read-only
`POST /workspaces/{id}/retrieve` probe instead; `POST /admin/eval/run` is not a
substitute (it reads only the packaged `seed/gold_qa.jsonl`, has no page gold,
and still calls `orch.answer()`).

**Direct-host experiment numbers are still not API numbers.**

## Steps 3–12: state as of 2026-10-19

The table below was written against an earlier tree. Four rows were wrong by the
time this round ended, so each now says what was actually re-checked.

| Item | State |
|---|---|
| `openai_compatible` / omniroute provider | **Fixed** — commit dbaaa7e. `get_llm()` has an `omniroute` branch (`providers/llm.py:484`) that constructs `OmnirouteLLM(OpenAIChatLLM)` (`:368`); the factory only returns `StubLLM` when the key is unset, and it logs a warning when it does. `openai_api_key`/`anthropic_api_key`/`*_chat_model` are declared on `Settings` as `SecretStr` and read with `get_secret_value()`. The silently-stubbed factory was the cause of the "0 answers" live battery. |
| Ollama / Anthropic zero-tool-call bug | **Still true.** `AnthropicLLM.chat` (`:395`) and `OllamaLLM.chat` (`:403`) accept `tools` and never put it on the wire, then return `tool_calls=[]` unconditionally (`:420`, `:450`). Only the OpenAI-shaped path (`:269-278`) parses tool calls. Not on the live path — `.env` uses `LLM_PROVIDER=omniroute` — but it is a latent trap for anyone who switches provider. |
| Calculation engine (`app/calc`) | Absent. `ls app/` has no `calc`; no provenance object anywhere. |
| Query route enum | Absent. Still `mode: str = "auto"` (`reasoning/orchestrator.py:47`) plus the `is_metric_lookup` boolean (`query_understanding.py:75`). |
| Silent filter-drop | **Partly fixed, and the row as written is stale.** The vector path no longer degrades: `_dense_search` raises `ProviderError` on embedding *and* search failure (`hybrid.py:611-624`) — there is no BM25-only fallback any more. The company relaxation still happens (`:347-351`) but is reported as `company_filter_relaxed` in `filter_diagnostics` alongside the `applied_filter` actually used. The **year** filter is still dropped silently when no corpus year matches (`query_understanding.py:86-104`); the only trace of it is the absence of `year` in `applied_filter`. None of these abstain. |
| Persistent BM25 | Absent — `BM25Okapi` is still built over the candidate set per query (`hybrid.py:634`). |
| Verifier | **Much of the row is stale.** `citation.py` now does a numeric-subset test plus per-amount **currency and scale binding inside one cited chunk** (`:239-285`, with the PIF `ª`-as-SAR glyph special case), prefers an LLM judge and falls back to token overlap (`:160-176`), and classifies SUPPORTED / PARTIAL / REJECTED at `_REJECT_RATIO = 0.5` (`:58`, `:117`). The orchestrator **does** fail closed: on PARTIAL/REJECTED it replaces the draft with a refusal and records `citation_verification_abstained` (`orchestrator.py:219-231`). Still true: no `Decimal` comparison, and `harness.py:_num_close` is float with 0.01 tolerance (`:242-246`). |
| Evaluation metrics | **Two harnesses, and they are not the same thing.** The packaged serving harness still has the placeholder pair — `context_precision` and `context_recall` are the identical `min(1.0, n_retrieved/5.0)` (`harness.py:190-191`). Recall@K now exists, but only in the **offline** stage-attribution harness (`evaluation/pif_stages.py`) and the experiment scripts. No Precision@K, MRR, NDCG, EM or F1 anywhere; FinQA/TAT-QA answer scoring absent. |
| FinQA / TAT-QA | Absent. |
| Market data | Absent. |
| Vision / chart module | Absent — `vision_model` is consumed only by the config's own role map (`config.py:125`, `:290`); no module passes an image to a model. `chart-reading` exists only as the probe below. |
| Frontend login | Absent — `ConnectionGate.tsx` is still a paste-API-key flow. |
| Auth holes | Still open, and now measurable: the **11 xfails** in `tests/test_api_isolation.py` are exactly these — unauthenticated `POST /auth/tenants`, `POST /auth/users` trusting a caller-supplied `tenant_id`, the API key echoed by `UserOut`/`/auth/me` and stored in plaintext, `GET /admin/eval/runs` with no tenant filter, and `POST /messages/{id}/feedback` with no workspace check. No login rate limiting. The prod `JWT_SECRET` guard is **correct** (`config.py`). |

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

### Embeddings and reranking: none on the gateway — corrected 2026-10-19

All 8 embedding candidates returned upstream `404`, and there is **no rerank
route**, so embeddings must come from a local model. The sentence *"Neither
option is currently installed on this machine"* written at that time is **no
longer true**, and contradicted the Decision-2 section above; the accurate state
is:

- **Embeddings work locally.** `ollama serve` was started and
  `nomic-embed-text` (274 MB) pulled to verify `OllamaEmbeddings` for real —
  768-dim, L2-normalised, `EMBEDDINGS_PROVIDER=ollama` is what `.env` runs
  today. Cold first call 32.2 s (model load).
- **Reranking does not.** `RERANKER_PROVIDER=stub`, `COHERE_API_KEY` empty, and
  `sentence_transformers` / `torch` / `cohere` are absent from the venv, so
  `CohereReranker` and the `cross-encoder/ms-marco-MiniLM-L-6-v2` path cannot be
  constructed. The live reranker is the query-term-coverage stub. This is the
  single most consequential fact in the retrieval findings below — the page
  quota's measured gain partly *compensates* for the stub.
- Ollama serves only `nomic-embed-text` and `llama3.2:3b`; no reranker model is
  available through it either.

## Atlas Search spike: VIABLE

`mongodb/mongodb-atlas-local` runs correctly on this Windows/Docker box —
**12/12 checks passed**. `$vectorSearch` with a `tenant_id` filter returned only
the requesting tenant's chunks (7 ms warm), `$search` full-text with a keyword
tenant filter worked (102 ms), `Decimal128` round-tripped, replica-set
transactions committed, RSS ~810 MiB. Two gotchas found: exact-match filters
need `lucene.keyword` (not `lucene.keyword_analyzer`), and the score metadata
field is `$meta: "vectorSearchScore"`.

## Running the app in Docker, and where the API key comes from — verified 2026-10-19

`start.bat` / `stop.bat` (repo root) wrap the `app` profile. Start is
`docker compose -f infra/docker-compose.yml --profile app up -d` followed by a
health poll on `finrag-backend` (180s cap) before the URLs are printed. Stop
takes down only `backend`/`frontend`; `mongo`, `mongo-test` and `qdrant` stay up
so the next start is instant.

The demo credentials have to be created **inside** the backend container:

```
docker exec finrag-backend python -m app.cli seed
```

Raw output (key truncated — it is a credential, not to be committed):

```
✓ created tenant: 2395931ada5f46db99c3c570acb3f921
✓ created user: analyst@demo.local
    email   : analyst@demo.local
    api_key : 2906…  (48 chars, redacted)
    workspace: 40d2e18542094a768a5a01b42a30b71d  (Demo Workspace)
```

`app.cli seed` reuses an existing tenant/user/workspace and always prints the
`api_key` (`cli.py:93-96`), so re-running it is credential recovery, not a
duplicate insert.

That key was read back from the container's own store and replayed against both
the API and the browser path, without ever printing it:

```
backend direct  200  [{"id":"40d2e18542094a768a5a01b42a30b71d","name":"Demo Workspace",…
via Next proxy  200  [{"id":"40d2e18542094a768a5a01b42a30b71d",…
auth/me proxy   200  {"id":"11e2669df2e84cb6a332851c46c5e7a2","email":"analyst@demo.local",…
bogus key via proxy 401
```

Why the host command no longer works for the container UI — the same
`select(User)` run in both places:

```
HOST       D:\Project\Agentic-Fin-Rag\backend\data\app.db   users: 0  workspaces: 0
CONTAINER  /app/data/app.db  (volume backend_data)          users: 2  workspaces: 2
```

Two separate SQLite files, two separate MongoDB servers (host `27017` native
mongod vs container `27019`). A key seeded on the host cannot authenticate the
Docker API. This was the reported symptom.

Smoke checks this round: `GET :3000/` → 200, `GET :3000/api/health` → 200
(through the Next rewrite), `GET :3000/api/workspaces` with no key → 401,
`GET :8010/docs` → 200. `stop.bat` then left both ports refusing (`HTTP 000`)
and `start.bat` restored them to healthy.

Side effect worth knowing: `finrag-mongo-test` (`:27018`) is a default-profile
service, so `--profile app up` starts it too. Harmless, but it is now always
running after `start.bat`.

Open question for the user: `demo.ps1` and the host path in `README.md` still
write to `backend/data/` (SQLite + Qdrant local mode). With the containers as
the default run mode, those host stores are dead weight — either keep both paths
documented (current state) or retire the host path.

