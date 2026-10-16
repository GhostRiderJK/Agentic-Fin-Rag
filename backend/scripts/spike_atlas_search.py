"""Atlas Vector Search / Atlas Search feasibility spike.

Answers one question: on this machine, does mongodb/mongodb-atlas-local give us
working $vectorSearch and $search, so we could retire Qdrant and the in-process
BM25 rebuild?

Usage:
    .venv/Scripts/python.exe backend/scripts/spike_atlas_search.py
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time

try:
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure, ServerSelectionTimeoutError
except ImportError:  # pragma: no cover
    sys.exit("pymongo not installed")

URL = "mongodb://127.0.0.1:27018/?directConnection=true"
DB = "finrag_spike"
COLL = "chunks"
DIM = 64
DOCS = 300

checks: list[dict] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"check": name, "ok": ok, "detail": detail[:400]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail[:280]}" if detail else ""))
    return ok


def ping_mongot(client: MongoClient) -> tuple[bool, str]:
    """Atlas Search needs the mongot process; ask the server what it sees.
    Uses the collection-level list API, which is a genuine mongot round-trip."""
    probe = client[DB]["_mongot_probe"]
    try:
        listed = list(probe.list_search_indexes())
        return True, f"list_search_indexes ok ({len(listed)} indexes)"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        return False, f"{type(exc).__name__}: {msg}"


def main() -> None:
    print(f"Connecting to {URL}\n")

    # 1. connectivity
    try:
        client = MongoClient(URL, serverSelectionTimeoutMS=8000)
        info = client.admin.command("serverStatus")
        record(
            "connectivity",
            True,
            f"version={info.get('version')} replicaSet={info.get('replica') and info['replica'].get('setName')}",
        )
    except ServerSelectionTimeoutError as exc:
        record("connectivity", False, str(exc))
        print("\nNo MongoDB reachable — start the atlas-local container first.")
        sys.exit(1)

    # 2. auth-free / capability
    version = str(info.get("version", ""))
    record("server_version_modern", tuple(int(p) for p in version.split(".")[:2]) >= (7, 0), f"v{version}")

    # 3. search subsystem present?
    mongot_ok, mongot_detail = ping_mongot(client)
    record("search_subsystem_available", mongot_ok, mongot_detail)

    # 4. seed documents with vectors + text
    random.seed(7)
    coll = client[DB][COLL]
    coll.drop()
    rows = [
        {
            "tenant_id": "t-a" if i % 2 == 0 else "t-b",
            "company": random.choice(["AAPL", "MSFT", "NVDA"]),
            "fiscal_year": random.choice([2022, 2023, 2024]),
            "text": (
                f"chunk {i}: "
                + random.choice(
                    [
                        "Apple reported total net sales of three hundred eighty three billion dollars.",
                        "Microsoft intelligent cloud operating margin expanded year over year.",
                        "Nvidia data center revenue grew sharply on demand for accelerated compute.",
                        "Gross margin, operating margin and free cash flow are reported separately.",
                        "The board approved a share repurchase programme of ten billion dollars.",
                    ]
                )
            ),
            "embedding": [round(random.random(), 4) for _ in range(DIM)],
        }
        for i in range(DOCS)
    ]
    t0 = time.perf_counter()
    coll.insert_many(rows)
    record(
        "bulk_insert",
        True,
        f"{DOCS} docs in {time.perf_counter() - t0:.2f}s, dim={DIM}",
    )

    # 5. create vector search index
    vector_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": DIM,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "tenant_id"},
            {"type": "filter", "path": "company"},
            {"type": "filter", "path": "fiscal_year"},
        ],
    }
    created_vector = False
    try:
        coll.create_search_index(
            {"name": "vs_idx", "type": "vectorSearch", "definition": vector_definition}
        )
        created_vector = True
        record("create_vector_search_index", True, "name=vs_idx")
    except Exception as exc:  # noqa: BLE001
        record("create_vector_search_index", False, f"{type(exc).__name__}: {exc}")

    # 6. create full-text search index
    created_text = False
    try:
        coll.create_search_index(
            {
                "name": "text_idx",
                "type": "search",
                "definition": {
                    "analyzer": "lucene.standard",
                    "searchAnalyzer": "lucene.standard",
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "text": {"type": "string", "analyzer": "lucene.standard"},
                            "tenant_id": {"type": "string", "analyzer": "lucene.keyword"},
                            "company": {"type": "string", "analyzer": "lucene.keyword"},
                        },
                    },
                },
            }
        )
        created_text = True
        record("create_fulltext_search_index", True, "name=text_idx")
    except Exception as exc:  # noqa: BLE001
        record("create_fulltext_search_index", False, f"{type(exc).__name__}: {exc}")

    # 7. wait for index build
    if created_vector or created_text:
        deadline = time.perf_counter() + 90
        status = "unknown"
        while time.perf_counter() < deadline:
            try:
                listed = list(coll.list_search_indexes())
                status = ", ".join(f"{i.get('name')}={i.get('status')}" for i in listed)
                if all(i.get("status") in {"READY", "STEADY"} for i in listed if i.get("name")):
                    break
            except Exception as exc:  # noqa: BLE001
                status = f"list failed: {exc}"
            time.sleep(3)
        record(
            "indexes_reach_STEADY",
            all(
                i.get("status") in {"READY", "STEADY"}
                for i in (list(coll.list_search_indexes()) if created_vector or created_text else [])
            )
            and "PENDING" not in status,
            status,
        )

    # 8. run $vectorSearch with tenant filter
    if created_vector:
        qvec = [round(random.random(), 4) for _ in range(DIM)]
        try:
            t0 = time.perf_counter()
            agg = list(
                coll.aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": "vs_idx",
                                "path": "embedding",
                                "queryVector": qvec,
                                "numCandidates": 100,
                                "limit": 5,
                                "filter": {"tenant_id": "t-a", "company": "AAPL"},
                            }
                        },
                        {"$project": {"_id": 0, "company": 1, "tenant_id": 1, "score": {"$meta": "vectorSearchScore"}}},
                    ]
                )
            )
            dt = time.perf_counter() - t0
            ok = len(agg) > 0 and all(r["tenant_id"] == "t-a" for r in agg)
            record(
                "vectorsearch_with_tenant_filter",
                ok,
                f"{len(agg)} hits in {dt * 1000:.0f}ms; top={json.dumps(agg[0] if agg else {})}",
            )
        except Exception as exc:  # noqa: BLE001
            record("vectorsearch_with_tenant_filter", False, f"{type(exc).__name__}: {exc}")

    # 9. run $search
    if created_text:
        try:
            t0 = time.perf_counter()
            agg = list(
                coll.aggregate(
                    [
                        {
                            "$search": {
                                "index": "text_idx",
                                "compound": {
                                    "must": [{"text": {"query": "operating margin revenue", "path": "text"}}],
                                    "filter": [{"term": {"query": "t-a", "path": "tenant_id"}}],
                                },
                            }
                        },
                        {"$limit": 5},
                        {"$project": {"_id": 0, "tenant_id": 1, "score": {"$meta": "searchScore"}, "text": {"$substrCP": ["$text", 0, 60]}}},
                    ]
                )
            )
            dt = time.perf_counter() - t0
            ok = len(agg) > 0 and all(r["tenant_id"] == "t-a" for r in agg)
            record("fulltext_search_with_tenant_filter", ok, f"{len(agg)} hits in {dt * 1000:.0f}ms")
        except Exception as exc:  # noqa: BLE001
            record("fulltext_search_with_tenant_filter", False, f"{type(exc).__name__}: {exc}")

    # 10. Decimal128 + transactions (the money-storage requirement)
    try:
        from bson.decimal128 import Decimal128

        c2 = client[DB]["decimals"]
        c2.insert_one({"amount": Decimal128("12345678901.234567890")})
        got = c2.find_one({"amount": {"$exists": True}})
        record("decimal128_roundtrip", str(got["amount"]) == "12345678901.234567890", str(got["amount"]))
    except Exception as exc:  # noqa: BLE001
        record("decimal128_roundtrip", False, f"{type(exc).__name__}: {exc}")

    try:
        client[DB]["txn_probe"].drop()
        with client.start_session() as session:
            with session.start_transaction():
                client[DB]["txn_probe"].insert_one({"v": 1}, session=session)
        record("transactions_supported", client[DB]["txn_probe"].count_documents({"v": 1}) == 1, "replica-set txn ok")
    except Exception as exc:  # noqa: BLE001
        record("transactions_supported", False, f"{type(exc).__name__}: {exc}")

    # 11. footprint
    if shutil.which("docker"):
        try:
            stats = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}}", "finrag-atlas"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            record("container_footprint", True, stats.stdout.strip() or stats.stderr.strip())
        except Exception as exc:  # noqa: BLE001
            record("container_footprint", False, str(exc))

    passed = sum(1 for c in checks if c["ok"])
    print(f"\n{passed}/{len(checks)} checks passed")
    verdict = all(c["ok"] for c in checks if c["check"] in {
        "connectivity", "search_subsystem_available", "vectorsearch_with_tenant_filter",
        "fulltext_search_with_tenant_filter", "indexes_reach_STEADY",
    })
    print("VERDICT:", "ATLAS CONSOLIDATION VIABLE" if verdict else "ATLAS NOT READY — STAY ON QDRANT")


if __name__ == "__main__":
    main()
