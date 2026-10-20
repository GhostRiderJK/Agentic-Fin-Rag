# Working rules (read every session)
- Never write "done", "added", "wired" or "all pass" without raw command output pasted in the reply. If something was not done, write NOT DONE.
- One task per prompt. Do exactly what is asked and stop.
- Windows PowerShell only. No bash syntax such as VAR=1 cmd.
- Use 127.0.0.1, never localhost. Never touch the finrag dev database in tests.
- Never print, log or commit secrets. Read them from environment variables only.
- Pin exact model IDs. Never use OmniRoute auto/* routing in evals.
- A test result only counts if it comes from plain `python -m pytest -q` run in backend. Never use --ignore, --deselect, -k or -x to produce a number.
- Every file you say you changed must appear in `git status --short`. Anything else is NOT DONE.
- Never delete, rename, weaken or merge an existing test or assertion. If a test must change, say so and wait.
- Never run git commit, stash, reset, checkout or clean unless asked.
- Do not create backup copies (*_old.py), logs, status files or summaries in the repo unless asked.

# Runtime
- The app runs in Docker: `start.bat` / `stop.bat` at the repo root (compose `app`
  profile). Frontend http://127.0.0.1:3000, API http://127.0.0.1:8010 (container
  port 8000), MongoDB 127.0.0.1:27019, Qdrant 6333.
- Demo credentials come from `docker exec finrag-backend python -m app.cli seed`.
  The host `python -m app.cli seed` writes a different SQLite file, so its key gets
  401 from the container API. Never paste a real key into a doc, log, or reply.
- Do not start host-side `ira serve` or `npm run dev` unless asked: port 3000 is
  already published by the frontend container, and the API container deliberately
  uses host 8010 so a host `ira serve` can keep 8000.

# Test environment
- Run tests from D:\Project\agentic-fin-rag\backend (pyproject.toml is there).
- Real MongoDB for integration tests: Atlas-local container `mongodb-atlas-test` at mongodb://127.0.0.1:27018/?directConnection=true (MONGODB_TEST_URL). Dev MongoDB (native service): 127.0.0.1:27017.
- Tests that need a real server skip themselves when it is unreachable. A run with skips does not prove the real-server cases.