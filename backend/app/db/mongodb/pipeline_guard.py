"""Tenant-safe aggregation pipelines.

``BaseRepository.aggregate`` used to prepend ``{"$match": {"tenant_id": ...}}``
and nothing else. That only scopes the *root* collection. Any stage that reads
another collection (``$lookup``, ``$graphLookup``, ``$unionWith``) or writes
(``$out``, ``$merge``) walks straight past it, and ``$function`` /
``$accumulator`` / ``$where`` run arbitrary JavaScript on the server.

This module turns a caller's pipeline into one that is provably scoped:

* Only stages on an allow-list are accepted. Anything else is rejected, which
  keeps ``$vectorSearch``, ``$search``, ``$geoNear``, ``$collStats`` and the
  like out until they get dedicated, reviewed repository methods.
* ``$out`` and ``$merge`` are never allowed, for any caller (super admin too).
* ``$function``, ``$accumulator`` and ``$where`` are rejected anywhere in a
  stage, however deeply nested.
* ``$lookup`` / ``$graphLookup`` / ``$unionWith`` may only read collections
  the caller explicitly allow-lists, and a tenant filter is injected on the
  *foreign* side:
    - ``$lookup``      pipeline form only; ``{"$match": tenant}`` becomes the
                       first stage of the foreign pipeline. The
                       ``localField`` / ``foreignField`` form cannot be
                       filtered and is rejected.
    - ``$graphLookup`` ``restrictSearchWithMatch`` is set/merged with the
                       tenant filter.
    - ``$unionWith``   rewritten to the document form with the tenant
                       ``$match`` as the first stage of its pipeline.
* ``$facet`` sub-pipelines are guarded recursively (they already see
  tenant-scoped input, so no extra ``$match`` is prepended to them).

Allow-listed foreign collections MUST carry a ``tenant_id`` field on every
document. Never allow-list a collection that does not (for example
``tenants``).

``tenant_id=None`` means "unscoped" (super admin): the root ``$match`` and the
foreign-side filters are skipped, but every ban and the allow-lists still
apply.

The guard is pure Python with no third-party imports so it can be unit-tested
without a database.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

__all__ = ["PipelineGuardError", "guard_pipeline"]

TENANT_FIELD = "tenant_id"

MAX_DEPTH = 6
MAX_TOTAL_STAGES = 100
_MAX_SCAN_NODES = 100_000

# Stages that only transform the documents flowing through them.
_PURE_STAGES = frozenset(
    {
        "$match",
        "$project",
        "$addFields",
        "$set",
        "$unset",
        "$group",
        "$sort",
        "$limit",
        "$skip",
        "$count",
        "$unwind",
        "$sortByCount",
        "$replaceRoot",
        "$replaceWith",
        "$bucket",
        "$bucketAuto",
        "$sample",
        "$redact",
        "$setWindowFields",
        "$fill",
        "$densify",
    }
)
_FOREIGN_STAGES = frozenset({"$lookup", "$graphLookup", "$unionWith"})
_NESTED_STAGES = frozenset({"$facet"})
_ALLOWED_STAGES = _PURE_STAGES | _FOREIGN_STAGES | _NESTED_STAGES

_WRITE_STAGES = frozenset({"$out", "$merge"})
_BANNED_OPERATORS = frozenset({"$function", "$accumulator", "$where"})

_LOOKUP_KEYS = frozenset({"from", "let", "pipeline", "as"})
_GRAPH_LOOKUP_KEYS = frozenset(
    {
        "from",
        "startWith",
        "connectFromField",
        "connectToField",
        "as",
        "maxDepth",
        "depthField",
        "restrictSearchWithMatch",
    }
)
_UNION_KEYS = frozenset({"coll", "pipeline"})


class PipelineGuardError(ValueError):
    """Raised when an aggregation pipeline cannot be proven tenant-safe."""


@dataclass
class _State:
    tenant_id: Optional[str]
    allowed: frozenset
    remaining: int = MAX_TOTAL_STAGES


def guard_pipeline(
    pipeline: Iterable[Mapping[str, Any]],
    *,
    tenant_id: Optional[str],
    allow_foreign_collections: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Return a tenant-scoped copy of ``pipeline`` or raise PipelineGuardError.

    The caller's pipeline is never mutated.
    """
    if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id):
        raise PipelineGuardError("tenant_id must be a non-empty string")
    if isinstance(allow_foreign_collections, (str, bytes)):
        raise PipelineGuardError(
            "allow_foreign_collections must be a collection of names, not a string"
        )
    state = _State(
        tenant_id=tenant_id,
        allowed=frozenset(allow_foreign_collections),
    )
    stages = _guard_stages(pipeline, state, depth=0)
    return _tenant_first(stages, tenant_id)


# -- helpers ---------------------------------------------------------------


def _tenant_first(
    stages: List[Dict[str, Any]], tenant_id: Optional[str]
) -> List[Dict[str, Any]]:
    if tenant_id is None:
        return stages
    return [{"$match": {TENANT_FIELD: tenant_id}}, *stages]


def _guard_stages(stages: Any, state: _State, depth: int) -> List[Dict[str, Any]]:
    if depth > MAX_DEPTH:
        raise PipelineGuardError("pipeline is nested too deeply")
    if isinstance(stages, (str, bytes, Mapping)) or not isinstance(stages, Iterable):
        raise PipelineGuardError("a pipeline must be a list of stages")
    return [_guard_stage(stage, state, depth) for stage in stages]


def _guard_stage(stage: Any, state: _State, depth: int) -> Dict[str, Any]:
    if not isinstance(stage, Mapping) or len(stage) != 1:
        raise PipelineGuardError(
            "each pipeline stage must be a mapping with exactly one operator"
        )
    state.remaining -= 1
    if state.remaining < 0:
        raise PipelineGuardError("pipeline has too many stages")

    ((op, body),) = stage.items()
    if op in _WRITE_STAGES:
        raise PipelineGuardError(f"{op} is never allowed: aggregation may not write")
    if op not in _ALLOWED_STAGES:
        raise PipelineGuardError(f"stage {op!r} is not allowed in tenant-scoped pipelines")

    _reject_banned_operators(body)

    if op == "$lookup":
        return {op: _guard_lookup(body, state, depth)}
    if op == "$graphLookup":
        return {op: _guard_graph_lookup(body, state)}
    if op == "$unionWith":
        return {op: _guard_union_with(body, state, depth)}
    if op == "$facet":
        return {op: _guard_facet(body, state, depth)}
    return {op: copy.deepcopy(body)}


def _reject_banned_operators(body: Any) -> None:
    """Reject code-execution operators anywhere inside a stage body.

    Iterative on purpose: a hostile, deeply nested document must not be able
    to trigger a RecursionError instead of a clean rejection.
    """
    stack = [body]
    seen = 0
    while stack:
        node = stack.pop()
        seen += 1
        if seen > _MAX_SCAN_NODES:
            raise PipelineGuardError("pipeline is too large to inspect")
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in _BANNED_OPERATORS:
                    raise PipelineGuardError(f"operator {key} is not allowed")
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def _check_foreign(name: Any, state: _State, op: str) -> str:
    if not isinstance(name, str) or not name:
        raise PipelineGuardError(f"{op} must name its collection with a plain string")
    if name not in state.allowed:
        raise PipelineGuardError(
            f"{op} collection {name!r} is not in the allowed foreign collections"
        )
    return name


def _check_keys(body: Mapping[str, Any], allowed: frozenset, op: str) -> None:
    extra = sorted(str(k) for k in body if k not in allowed)
    if extra:
        raise PipelineGuardError(f"{op} has unsupported option(s): {', '.join(extra)}")


def _guard_lookup(body: Any, state: _State, depth: int) -> Dict[str, Any]:
    if not isinstance(body, Mapping):
        raise PipelineGuardError("$lookup body must be a mapping")
    if "localField" in body or "foreignField" in body:
        raise PipelineGuardError(
            "$lookup localField/foreignField cannot be tenant-scoped; "
            "use the pipeline form"
        )
    _check_keys(body, _LOOKUP_KEYS, "$lookup")
    if "pipeline" not in body:
        raise PipelineGuardError("$lookup requires the pipeline form to be tenant-scoped")
    _check_foreign(body.get("from"), state, "$lookup")

    guarded = {k: copy.deepcopy(v) for k, v in body.items() if k != "pipeline"}
    sub = _guard_stages(body["pipeline"], state, depth + 1)
    guarded["pipeline"] = _tenant_first(sub, state.tenant_id)
    return guarded


def _guard_graph_lookup(body: Any, state: _State) -> Dict[str, Any]:
    if not isinstance(body, Mapping):
        raise PipelineGuardError("$graphLookup body must be a mapping")
    _check_keys(body, _GRAPH_LOOKUP_KEYS, "$graphLookup")
    _check_foreign(body.get("from"), state, "$graphLookup")

    guarded = copy.deepcopy(dict(body))
    if state.tenant_id is None:
        return guarded
    tenant_filter = {TENANT_FIELD: state.tenant_id}
    existing = body.get("restrictSearchWithMatch")
    if existing is None:
        guarded["restrictSearchWithMatch"] = tenant_filter
    elif isinstance(existing, Mapping):
        guarded["restrictSearchWithMatch"] = {
            "$and": [copy.deepcopy(dict(existing)), tenant_filter]
        }
    else:
        raise PipelineGuardError("$graphLookup restrictSearchWithMatch must be a mapping")
    return guarded


def _guard_union_with(body: Any, state: _State, depth: int) -> Dict[str, Any]:
    if isinstance(body, str):
        coll: Any = body
        pipeline: Any = []
    elif isinstance(body, Mapping):
        _check_keys(body, _UNION_KEYS, "$unionWith")
        coll = body.get("coll")
        pipeline = body.get("pipeline", [])
        if coll is None:
            raise PipelineGuardError("$unionWith requires a collection name")
    else:
        raise PipelineGuardError("$unionWith body must be a collection name or a mapping")
    _check_foreign(coll, state, "$unionWith")

    sub = _guard_stages(pipeline, state, depth + 1)
    return {"coll": coll, "pipeline": _tenant_first(sub, state.tenant_id)}


def _guard_facet(body: Any, state: _State, depth: int) -> Dict[str, Any]:
    if not isinstance(body, Mapping) or not body:
        raise PipelineGuardError("$facet requires at least one named sub-pipeline")
    return {
        name: _guard_stages(sub, state, depth + 1) for name, sub in body.items()
    }
