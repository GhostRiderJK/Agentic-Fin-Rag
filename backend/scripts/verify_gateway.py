"""Gateway capability probe for FinRAG 2.0.

Runs tiny real requests against an OpenAI-compatible gateway and records, per
model, which capabilities actually work. Nothing here is mocked: a PASS means a
2xx response with a well-formed body.

Usage (PowerShell):
    $env:LLM_API_KEY = "<key>"        # or keep it in .env
    .venv/Scripts/python.exe backend/scripts/verify_gateway.py --role chat
    .venv/Scripts/python.exe backend/scripts/verify_gateway.py --all

The API key is read from the environment or .env and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "http://localhost:20128/v1"

# 1x1 red pixel, keeps the vision probe tiny and dependency-free.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# Header names that hint at which upstream the gateway actually used.
PROVIDER_HINT_HEADERS = (
    "x-provider",
    "x-upstream",
    "x-omniroute-provider",
    "x-request-id",
    "x-model",
    "x-account",
    "x-retry-count",
    "x-ratelimit-remaining",
    "x-ratelimit-limit",
    "retry-after",
)


def load_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=value lines. Values are taken up to the first whitespace, so
    inline `# comment` tails in .env do not become part of the value."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip().split("#")[0].strip()
        value = value.strip("'\"")
        out[key] = value
    return out


def resolve_config(env_file: Path) -> tuple[str, str]:
    file_env = load_env_file(env_file)

    def pick(*names: str) -> str | None:
        for name in names:
            value = os.environ.get(name) or file_env.get(name)
            if value:
                return value
        return None

    base_url = (pick("LLM_BASE_URL", "OMNIROUTE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    api_key = pick("LLM_API_KEY", "OMNIROUTE_API_KEY")
    if not api_key:
        sys.exit(
            "No API key found. Set LLM_API_KEY (or OMNIROUTE_API_KEY) in the "
            "environment or in .env. Aborting rather than mocking the gateway."
        )
    return base_url, api_key


def shortlist(models: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group nvidia/ models into probe buckets from catalog metadata + naming."""
    nv = [m for m in models if str(m.get("id", "")).startswith("nvidia/")]
    by_id = {m["id"]: m for m in nv}

    def ids(pred) -> list[str]:
        return [m["id"] for m in nv if pred(m)]

    def is_non_generation(m: dict) -> bool:
        return m.get("type") in {"image", "audio"} or any(
            tok in m["id"] for tok in ("flux", "whisper", "parakeet", "fastpitch", "tacotron", "nvclip")
        )

    def is_chat(m: dict) -> bool:
        if is_non_generation(m) or m.get("type") in {"embedding", "rerank"}:
            return False
        if m.get("input_modalities"):
            return "text" in m["input_modalities"]
        return True

    def has_image_input(m: dict) -> bool:
        return "image" in (m.get("input_modalities") or [])

    buckets = {
        "chat": ids(is_chat),
        "vision": ids(lambda m: is_chat(m) and has_image_input(m)),
        "embedding": ids(lambda m: m.get("type") == "embedding" or "embed" in m["id"]),
        "rerank": ids(lambda m: m.get("type") == "rerank" or "rerank" in m["id"]),
    }
    buckets["embedding"] = [i for i in buckets["embedding"] if i in by_id]
    return buckets


def _headers_hint(headers: httpx.Headers) -> dict[str, str]:
    return {k: headers[k] for k in PROVIDER_HINT_HEADERS if k in headers}


async def probe_chat(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: FINRAG_OK"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    r = await client.post("/chat/completions", json=payload)
    body = _json(r)
    text = _extract_text(body)
    ok = r.status_code == 200 and bool(text)
    return {
        "ok": ok,
        "status": r.status_code,
        "returned_model": body.get("model") if isinstance(body, dict) else None,
        "text": (text or "")[:80],
        "usage": (body.get("usage") if isinstance(body, dict) else None),
        "error": None if ok else _err(r, body),
        "hints": _headers_hint(r.headers),
    }


async def probe_tools(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is 1234.56 times 7? You must call the multiply tool.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 128,
        "temperature": 0,
    }
    r = await client.post("/chat/completions", json=payload)
    body = _json(r)
    calls = _extract_tool_calls(body)
    ok = r.status_code == 200 and bool(calls)
    return {
        "ok": ok,
        "status": r.status_code,
        "tool_calls": calls,
        "error": None if ok else _err(r, body) or "no tool_calls returned",
    }


async def probe_json(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": 'Return JSON only: {"company": "Apple", "revenue_usd_m": 9000}',
            }
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 64,
        "temperature": 0,
    }
    r = await client.post("/chat/completions", json=payload)
    body = _json(r)
    text = _extract_text(body)
    parsed = None
    if text:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
    ok = r.status_code == 200 and isinstance(parsed, dict)
    return {
        "ok": ok,
        "status": r.status_code,
        "json_parsed": ok,
        "text": (text or "")[:80],
        "error": None if ok else _err(r, body) or "response was not valid JSON",
    }


async def probe_stream(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "stream": True,
        "max_tokens": 32,
    }
    chunks = 0
    pieces: list[str] = []
    first_status = None
    err = None
    try:
        async with client.stream("POST", "/chat/completions", json=payload) as r:
            first_status = r.status_code
            if r.status_code != 200:
                raw = await r.aread()
                return {"ok": False, "status": r.status_code, "error": raw[:200].decode("utf-8", "replace")}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunks += 1
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # Reasoning models stream reasoning_content instead of content.
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        pieces.append(piece)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    text = "".join(pieces)
    return {
        "ok": first_status == 200 and chunks > 1 and bool(text) and err is None,
        "status": first_status,
        "chunks": chunks,
        "error": err,
    }


async def probe_vision(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this pixel?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"},
                    },
                ],
            }
        ],
        "max_tokens": 24,
        "temperature": 0,
    }
    r = await client.post("/chat/completions", json=payload)
    body = _json(r)
    text = _extract_text(body)
    ok = r.status_code == 200 and bool(text)
    return {
        "ok": ok,
        "status": r.status_code,
        "text": (text or "")[:60],
        "error": None if ok else _err(r, body),
    }


async def probe_embeddings(client: httpx.AsyncClient, model: str) -> dict:
    # input_type is an NVIDIA NIM extension; OpenAI-compatible gateways ignore it.
    r = await client.post(
        "/embeddings",
        json={"model": model, "input": ["revenue growth margin"], "input_type": "passage"},
    )
    body = _json(r)
    vec = None
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list) and data:
            emb = data[0].get("embedding") if isinstance(data[0], dict) else None
            if isinstance(emb, list):
                vec = emb
    ok = r.status_code == 200 and vec is not None
    return {
        "ok": ok,
        "status": r.status_code,
        "dim": len(vec) if vec else None,
        "usage": body.get("usage") if isinstance(body, dict) else None,
        "error": None if ok else _err(r, body),
    }


async def probe_rerank(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "query": "What was Apple revenue in 2023?",
        "documents": [
            "Apple reported annual revenue of 383 billion dollars in fiscal 2023.",
            "Bananas are a good source of potassium.",
        ],
        "top_n": 2,
    }
    # NVIDIA NIM reranks use POST /rerank (no /v1 prefix); OpenAI-style clients
    # sometimes expose /v1/rerank. Try both.
    for path in ("/rerank", "/v1/rerank", "/rerank/"):
        try:
            r = await client.post(path, json=payload)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code == 404:
            continue
        body = _json(r)
        results = None
        if isinstance(body, dict):
            results = body.get("results") or body.get("data")
        ok = r.status_code == 200 and isinstance(results, list) and len(results) > 0
        return {
            "ok": ok,
            "status": r.status_code,
            "path": path,
            "results": len(results) if isinstance(results, list) else None,
            "error": None if ok else _err(r, body),
        }
    return {"ok": False, "status": 404, "error": "no rerank route found"}


PROBES = {
    "chat": probe_chat,
    "tools": probe_tools,
    "json": probe_json,
    "stream": probe_stream,
    "vision": probe_vision,
    "embeddings": probe_embeddings,
    "rerank": probe_rerank,
}


def _json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def _extract_text(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):  # some gateways return content blocks
            return "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            ) or None
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    return None


def _extract_tool_calls(body: Any) -> list[dict]:
    if not isinstance(body, dict):
        return []
    choices = body.get("choices") or []
    if not choices:
        return []
    calls = (choices[0].get("message") or {}).get("tool_calls") or []
    out = []
    for c in calls:
        fn = c.get("function") or {}
        out.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return out


def _err(r: httpx.Response, body: Any) -> str | None:
    if r.status_code == 200:
        return None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return f"{r.status_code}: {err.get('code')} {str(err.get('message'))[:140]}"
        if isinstance(err, str):
            return f"{r.status_code}: {err[:140]}"
    return f"{r.status_code}: {r.text[:140]}"


async def run_model(
    client: httpx.AsyncClient, model: str, caps: list[str], sem: asyncio.Semaphore
) -> dict:
    results: dict[str, dict] = {}
    for cap in caps:
        async with sem:
            started = time.perf_counter()
            try:
                results[cap] = await PROBES[cap](client, model)
            except Exception as exc:  # noqa: BLE001
                results[cap] = {
                    "ok": False,
                    "status": None,
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                }
            results[cap]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        await asyncio.sleep(0.1)
    passed = [c for c, r in results.items() if r.get("ok")]
    return {"model": model, "results": results, "passed": passed}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    ap.add_argument("--models", help="comma-separated model ids to probe")
    ap.add_argument("--role", choices=["chat", "vision", "embedding", "rerank"])
    ap.add_argument(
        "--caps",
        help="comma-separated capability filter (chat,tools,json,stream,vision,embeddings,rerank)",
    )
    ap.add_argument("--limit", type=int, default=12, help="max models per role")
    ap.add_argument("--all", action="store_true", help="probe every nvidia/ model")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "gateway-verification.json"))
    args = ap.parse_args()

    base_url, api_key = resolve_config(Path(args.env_file))
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=args.timeout,
    ) as client:
        try:
            r = await client.get("/models")
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"GATEWAY UNREACHABLE at {base_url}: {exc}")
        if r.status_code != 200:
            sys.exit(f"GATEWAY /models returned HTTP {r.status_code}: {r.text[:200]}")
        models = r.json().get("data", [])

    nv = sorted(m["id"] for m in models if str(m.get("id", "")).startswith("nvidia/"))
    buckets = shortlist(models)

    if args.models:
        targets = args.models.split(",")
    elif args.role:
        targets = buckets[args.role][: args.limit]
    elif args.all:
        targets = nv
    else:
        # Default: a representative shortlist across roles.
        targets = (
            buckets["chat"][: args.limit]
            + buckets["vision"][: args.limit]
            + buckets["embedding"]
            + buckets["rerank"]
        )

    seen: set[str] = set()
    ordered = [m for m in targets if not (m in seen or seen.add(m))]

    caps_filter = set(args.caps.split(",")) if args.caps else None

    metadata = {m["id"]: m for m in models if m.get("id") in set(ordered)}

    async def build_plan(model: str) -> list[str]:
        meta = metadata.get(model, {})
        caps: list[str] = []
        if meta.get("type") == "embedding" or "embed" in model:
            caps = ["embeddings"]
        elif meta.get("type") == "rerank" or "rerank" in model:
            caps = ["rerank"]
        elif meta.get("type") in {"image", "audio"}:
            return []
        else:
            caps = ["chat", "tools", "json", "stream"]
            if "image" in (meta.get("input_modalities") or []):
                caps.append("vision")
        if caps_filter:
            caps = [c for c in caps if c in caps_filter]
        return caps

    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=args.timeout,
        follow_redirects=True,
    ) as client:
        print(f"Gateway: {base_url}")
        print(f"nvidia/ models exposed: {len(nv)}   probing: {len(ordered)}\n")
        tasks = []
        for model in ordered:
            caps = await build_plan(model)
            if caps:
                tasks.append(run_model(client, model, caps, sem))
        done = await asyncio.gather(*tasks)

    results = []
    for entry in done:
        meta = metadata.get(entry["model"], {})
        entry["declared"] = {
            "type": meta.get("type"),
            "context_length": meta.get("context_length"),
            "dimensions": meta.get("dimensions"),
            "input_modalities": meta.get("input_modalities"),
            "capabilities": meta.get("capabilities"),
        }
        results.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "nvidia_model_count": len(nv),
                "buckets": buckets,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    header = f"{'MODEL':<62} {'PASS':<26} {'CHAT_MS':>8}"
    print(header)
    print("-" * len(header))
    for entry in results:
        marks = " ".join(
            f"{c}:{'Y' if entry['results'][c].get('ok') else 'n'}" for c in entry["results"]
        )
        latency = entry["results"].get("chat", {}).get("latency_ms", "")
        print(f"{entry['model']:<62} {marks:<26} {str(latency):>8}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
