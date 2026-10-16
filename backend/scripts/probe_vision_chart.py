"""Vision probe: read real numbers out of a generated chart image.

verify_gateway.py's vision check only proves a model accepts an image. This
script answers the question that matters for FinRAG: can it read *values* from
a financial chart, and are those values correct?

The chart is drawn with PIL from an explicit ground-truth table, so the expected
answer is known exactly and the image is byte-identical between runs.

Usage (PowerShell):
    $env:LLM_API_KEY = "<key>"        # or keep it in .env
    .venv/Scripts/python.exe backend/scripts/probe_vision_chart.py
    .venv/Scripts/python.exe backend/scripts/probe_vision_chart.py ^
        --models nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b --seed 7

The API key is read from the environment or .env and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_gateway import _err, _extract_text, _headers_hint, resolve_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ground truth for the generated chart. Values are USD millions.
SERIES = {"Q1": 1240.0, "Q2": 1585.0, "Q3": 1372.0, "Q4": 1908.0}

PROMPT = (
    "This bar chart shows quarterly revenue in USD millions. Read the exact "
    "value of each bar from the axis — do not estimate from bar height alone. "
    'Answer with JSON only: {"Q1": <number>, "Q2": <number>, "Q3": <number>, '
    '"Q4": <number>}'
)

AXIS_STEP = 200.0


def build_chart_png() -> bytes:
    """Deterministic bar chart of SERIES. Requires no network and no RNG."""
    from PIL import Image, ImageDraw

    w, h = 900, 600
    left, right, top, bottom = 110, 40, 70, 90
    plot_w, plot_h = w - left - right, h - top - bottom
    vmax = 2000.0

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=20)
        big = ImageFont.load_default(size=28)
    except Exception:  # older Pillow
        from PIL import ImageFont

        font = ImageFont.load_default()
        big = font

    # horizontal gridlines + axis labels
    for step in range(0, int(vmax) + 1, int(AXIS_STEP)):
        y = top + plot_h - (step / vmax) * plot_h
        d.line([(left, y), (left + plot_w, y)], fill=(220, 220, 220), width=1)
        d.text((left - 8 - 46, y - 8), f"{step:,}", fill=(60, 60, 60), font=font)

    d.line([(left, top), (left, top + plot_h)], fill=(30, 30, 30), width=2)
    d.line([(left, top + plot_h), (left + plot_w, top + plot_h)],
           fill=(30, 30, 30), width=2)

    labels = list(SERIES)
    n = len(labels)
    slot = plot_w / n
    bar_w = slot * 0.55
    for i, label in enumerate(labels):
        value = SERIES[label]
        bh = (value / vmax) * plot_h
        x0 = left + i * slot + (slot - bar_w) / 2
        y0 = top + plot_h - bh
        d.rectangle([x0, y0, x0 + bar_w, top + plot_h], fill=(46, 92, 176))
        d.text((x0 + bar_w / 2 - 14, top + plot_h + 12), label,
               fill=(30, 30, 30), font=font)

    d.text((left, 18), "Quarterly Revenue (USD millions)", fill=(20, 20, 20), font=big)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def parse_numbers(text: str) -> dict[str, float] | None:
    """Pull Q1..Q4 out of a reply, preferring a JSON object when present."""
    for candidate in (text, *re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and any(k.upper() in obj for k in SERIES):
            out: dict[str, float] = {}
            for key in SERIES:
                for k, v in obj.items():
                    if k.strip().upper() == key:
                        try:
                            out[key] = float(v)
                        except (TypeError, ValueError):
                            pass
            if out:
                return out
    out = {}
    for key, value in re.findall(r"\b(Q[1-4])\D{0,12}?(-?\d[\d,]*\.?\d*)", text, re.I):
        try:
            out.setdefault(key.upper(), float(value.replace(",", "")))
        except ValueError:
            continue
    return out or None


def score(parsed: dict[str, float] | None) -> dict[str, Any]:
    if not parsed:
        return {"correct": 0, "of": len(SERIES), "max_abs_err": None,
                "max_rel_err_pct": None, "exact": False}
    errs = {k: abs(parsed[k] - SERIES[k]) for k in SERIES if k in parsed}
    if not errs:
        return {"correct": 0, "of": len(SERIES), "max_abs_err": None,
                "max_rel_err_pct": None, "exact": False}
    worst = max(errs, key=lambda k: errs[k])
    rel = errs[worst] / SERIES[worst] * 100.0
    return {
        "correct": sum(1 for k in SERIES if errs.get(k) == 0.0),
        "of": len(SERIES),
        "max_abs_err": errs[worst],
        "worst_quarter": worst,
        "max_rel_err_pct": round(rel, 3),
        "exact": all(v == 0.0 for v in errs.values()),
        "missing": [k for k in SERIES if k not in parsed],
    }


async def probe_one(client: httpx.AsyncClient, model: str, image_b64: str,
                    seed: int | None, timeout: float,
                    max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    started = time.perf_counter()
    try:
        r = await client.post("/chat/completions", json=payload,
                              timeout=timeout)
    except Exception as e:
        return {"model": model, "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "latency_ms": round((time.perf_counter() - started) * 1000)}
    latency = round((time.perf_counter() - started) * 1000)

    body: Any = None
    try:
        body = r.json()
    except Exception:
        body = r.text[:400]

    problem = _err(r, body)
    if problem:
        return {"model": model, "status": "fail", "http_status": r.status_code,
                "error": problem, "latency_ms": latency,
                "headers": _headers_hint(r.headers)}

    text = _extract_text(body) or ""
    parsed = parse_numbers(text)
    finish = None
    try:
        finish = (body or {})["choices"][0]["finish_reason"]
    except Exception:
        pass
    return {
        "model": model,
        "status": "ok",
        "http_status": r.status_code,
        "latency_ms": latency,
        "max_tokens": max_tokens,
        "finish_reason": finish,
        "usage": (body or {}).get("usage") if isinstance(body, dict) else None,
        "headers": _headers_hint(r.headers),
        "reply": text[:1200],
        "parsed": parsed,
        "ground_truth": SERIES,
        **score(parsed),
    }


async def run(models: list[str], image_bytes: bytes, seed: int | None,
              timeout: float, env_file: Path, base_url_override: str | None,
              max_tokens: int) -> dict:
    base_url, api_key = resolve_config(env_file)
    if base_url_override:
        base_url = base_url_override
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        follow_redirects=True,
    ) as client:
        results = [await probe_one(client, m, image_b64, seed, timeout, max_tokens)
                   for m in models]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gateway_base_url": base_url,
        "seed": seed,
        "max_tokens": max_tokens,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_bytes": len(image_bytes),
        "prompt": PROMPT,
        "ground_truth": SERIES,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join([
        "nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/meta/llama-3.2-11b-vision-instruct",
    ]), help="comma-separated model ids")
    ap.add_argument("--image", default=None,
                    help="use this PNG/JPEG instead of the generated chart; "
                         "the score is then informational only, since ground "
                         "truth stays SERIES")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="reasoning models spend most of this on hidden "
                         "thoughts; too small yields a 2xx with no answer")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "vision-chart-probe.json"))
    args = ap.parse_args()

    image_bytes = (Path(args.image).read_bytes() if args.image
                   else build_chart_png())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    async def _go():
        return await run(models, image_bytes, args.seed, args.timeout,
                         Path(args.env_file), args.base_url, args.max_tokens)

    report = asyncio.run(_go())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"image sha256={report['image_sha256'][:16]}… "
          f"bytes={report['image_bytes']} seed={report['seed']}")
    print(f"ground truth: {SERIES}\n")
    for r in report["results"]:
        if r["status"] != "ok":
            print(f"{r['model']}\n  {r['status'].upper()} {r.get('http_status', '')} "
                  f"{r['error']} ({r['latency_ms']} ms)\n")
            continue
        print(f"{r['model']}")
        print(f"  ok in {r['latency_ms']} ms, finish={r['finish_reason']} "
              f"max_tokens={r['max_tokens']} usage={r['usage']}")
        print(f"  parsed: {r['parsed']}")
        print(f"  exact={r['exact']} correct={r['correct']}/{r['of']} "
              f"max_abs_err={r['max_abs_err']} max_rel_err={r['max_rel_err_pct']}%")
        print(f"  reply: {r['reply'][:300]}\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
