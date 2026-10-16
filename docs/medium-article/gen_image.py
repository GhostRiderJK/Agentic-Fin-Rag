"""Helper to generate images via Gemini 3.1 Flash Image API.

Usage:
    python gen_image.py <output_filename.png> "<prompt>"

Saves PNG into ./assets/ next to this script.
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

API_KEY = "AIzaSyD5JHKFWzJffKb72-yQkvRDIFKbEEyxvMo"
MODEL = "gemini-3.1-flash-image"
ASSETS = Path(__file__).parent / "assets"


def generate(prompt: str, out_path: Path) -> None:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={API_KEY}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Extract the first inline image part.
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData")
            if not inline:
                continue
            mime = inline.get("mimeType", "image/png")
            ext = "png" if "png" in mime else ("jpg" if "jpeg" in mime else "png")
            target = out_path.with_suffix("." + ext)
            target.write_bytes(base64.b64decode(inline["data"]))
            print(f"wrote {target} ({target.stat().st_size} bytes)")
            return
    print("NO IMAGE RETURNED. Raw response:")
    print(json.dumps(data, indent=2)[:2000])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_image.py <name> <prompt>")
        sys.exit(1)
    name, prompt = sys.argv[1], sys.argv[2]
    ASSETS.mkdir(parents=True, exist_ok=True)
    generate(prompt, ASSETS / name)
