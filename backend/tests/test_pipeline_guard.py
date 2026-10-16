"""Unit tests for app.db.mongodb.pipeline_guard.

Pure Python: no database, no fixtures. Each hostile pipeline here is one a
caller (or a prompt-injected agent) could plausibly build to read or write
across tenants.
"""

from __future__ import annotations

import copy

import pytest

from app.db.mongodb.pipeline_guard import (
    MAX_DEPTH,
    MAX_TOTAL_STAGES,
    PipelineGuardError,
    guard_pipeline,
)

T = "tenant-a"
ALLOW = ("chunks", "nodes")
TENANT_MATCH = {"$match": {"tenant_id": T}}


def guard(pipeline, tenant_id=T, allow=ALLOW):
    return guard_pipeline(
        pipeline, tenant_id=tenant_id, allow_foreign_collections=allow
    )


def lookup(frm="chunks", pipeline=None, **extra):
    body = {
        "from": frm,
        "let": {"d": "$doc_id"},
        "pipeline": pipeline
        if pipeline is not None
        else [{"$match": {"$expr": {"$eq": ["$doc_id", "$$d"]}}}],
        "as": "joined",
    }
    body.update(extra)
    return {"$lookup": body}


# -- root scoping -----------------------------------------------------------


def test_prepends_tenant_match_first():
    out = guard([{"$group": {"_id": "$kind", "n": {"$sum": 1}}}])
    assert out[0] == TENANT_MATCH
    assert len(out) == 2


def test_empty_pipeline_is_still_scoped():
    assert guard([]) == [TENANT_MATCH]


def test_callers_own_tenant_match_cannot_displace_the_root_match():
    hostile = {"$match": {"tenant_id": "tenant-b"}}
    out = guard([hostile])
    assert out[0] == TENANT_MATCH
    assert out[1] == hostile  # only ever narrows an already-scoped stream


def test_input_is_not_mutated_and_output_does_not_alias_it():
    stage = {"$match": {"status": "open"}}
    original = copy.deepcopy([stage])
    out = guard([stage])
    assert [stage] == original
    assert out[1]["$match"] is not stage["$match"]


# -- stages that are never allowed -----------------------------------------


@pytest.mark.parametrize("tenant_id", [T, None])
@pytest.mark.parametrize("op", ["$out", "$merge"])
def test_write_stages_are_never_allowed_even_for_super_admin(op, tenant_id):
    with pytest.raises(PipelineGuardError, match="never allowed"):
        guard([{op: "stolen"}], tenant_id=tenant_id)


@pytest.mark.parametrize(
    "op",
    [
        "$collStats",
        "$indexStats",
        "$currentOp",
        "$listSessions",
        "$listLocalSessions",
        "$planCacheStats",
        "$vectorSearch",
        "$search",
        "$searchMeta",
        "$geoNear",
        "$documents",
        "$changeStream",
    ],
)
def test_unlisted_stages_are_rejected(op):
    with pytest.raises(PipelineGuardError, match="not allowed"):
        guard([{op: {}}])


@pytest.mark.parametrize(
    "pipeline, operator",
    [
        ([{"$project": {"x": {"$function": {"body": "x", "args": [], "lang": "js"}}}}], "$function"),
        ([{"$group": {"_id": None, "x": {"$accumulator": {}}}}], "$accumulator"),
        ([{"$match": {"$where": "sleep(1000)"}}], "$where"),
        ([{"$facet": {"f": [{"$match": {"$where": "1"}}]}}], "$where"),
        ([lookup(let={"d": {"$function": {}}})], "$function"),
        ([{"$addFields": {"a": [[{"$function": {}}]]}}], "$function"),
    ],
)
def test_code_execution_operators_are_rejected_anywhere(pipeline, operator):
    with pytest.raises(PipelineGuardError, match="not allowed"):
        guard(pipeline)


@pytest.mark.parametrize(
    "bad",
    [
        {"$match": {}, "$limit": 1},
        {},
        "x",
        ["$match"],
        None,
    ],
)
def test_malformed_stages_are_rejected(bad):
    with pytest.raises(PipelineGuardError, match="exactly one operator"):
        guard([bad])


@pytest.mark.parametrize("bad", ["not a pipeline", {"$match": {}}, 5])
def test_pipeline_must_be_a_list_of_stages(bad):
    with pytest.raises(PipelineGuardError, match="list of stages"):
        guard(bad)


# -- $lookup ----------------------------------------------------------------


def test_lookup_simple_form_is_rejected():
    stage = {
        "$lookup": {
            "from": "chunks",
            "localField": "doc_id",
            "foreignField": "doc_id",
            "as": "joined",
        }
    }
    with pytest.raises(PipelineGuardError, match="pipeline form"):
        guard([stage])


def test_lookup_localfield_mixed_with_pipeline_is_rejected():
    with pytest.raises(PipelineGuardError, match="pipeline form"):
        guard([lookup(localField="doc_id", foreignField="doc_id")])


def test_lookup_requires_the_pipeline_key():
    with pytest.raises(PipelineGuardError, match="pipeline form"):
        guard([{"$lookup": {"from": "chunks", "as": "joined"}}])


def test_lookup_of_unlisted_collection_is_rejected():
    with pytest.raises(PipelineGuardError, match="allowed foreign collections"):
        guard([lookup(frm="users")])


def test_lookup_is_rejected_when_no_foreign_collections_are_allowed():
    with pytest.raises(PipelineGuardError, match="allowed foreign collections"):
        guard([lookup()], allow=())


@pytest.mark.parametrize(
    "frm", [{"db": "other", "coll": "chunks"}, None, "", 5, ["chunks"]]
)
def test_lookup_from_must_be_a_plain_string(frm):
    with pytest.raises(PipelineGuardError, match="plain string"):
        guard([lookup(frm=frm)])


def test_lookup_rejects_unknown_options():
    with pytest.raises(PipelineGuardError, match="unsupported"):
        guard([lookup(sneaky=1)])


def test_lookup_gets_a_tenant_match_as_the_first_foreign_stage():
    inner = {"$match": {"$expr": {"$eq": ["$doc_id", "$$d"]}}}
    out = guard([lookup(pipeline=[inner])])
    foreign = out[1]["$lookup"]["pipeline"]
    assert foreign[0] == TENANT_MATCH
    assert foreign[1] == inner


def test_nested_lookup_inside_a_lookup_is_also_scoped():
    nested = lookup(frm="nodes", pipeline=[{"$limit": 1}])
    out = guard([lookup(pipeline=[nested])])
    outer_pipeline = out[1]["$lookup"]["pipeline"]
    assert outer_pipeline[0] == TENANT_MATCH
    assert outer_pipeline[1]["$lookup"]["pipeline"][0] == TENANT_MATCH


def test_lookup_inside_facet_is_scoped():
    out = guard([{"$facet": {"f": [lookup()]}}])
    sub = out[1]["$facet"]["f"]
    assert sub[0]["$lookup"]["pipeline"][0] == TENANT_MATCH


def test_write_stage_hidden_in_a_lookup_pipeline_is_rejected():
    with pytest.raises(PipelineGuardError, match="never allowed"):
        guard([lookup(pipeline=[{"$out": "stolen"}])])


# -- $graphLookup -----------------------------------------------------------


def graph(**extra):
    body = {
        "from": "nodes",
        "startWith": "$doc_id",
        "connectFromField": "parent",
        "connectToField": "id",
        "as": "chain",
    }
    body.update(extra)
    return {"$graphLookup": body}


def test_graph_lookup_gets_a_tenant_restriction():
    out = guard([graph()])
    assert out[1]["$graphLookup"]["restrictSearchWithMatch"] == {"tenant_id": T}


def test_graph_lookup_merges_an_existing_restriction_without_losing_it():
    out = guard([graph(restrictSearchWithMatch={"kind": "folder"})])
    assert out[1]["$graphLookup"]["restrictSearchWithMatch"] == {
        "$and": [{"kind": "folder"}, {"tenant_id": T}]
    }


def test_graph_lookup_cannot_loosen_the_restriction_with_its_own_tenant_id():
    out = guard([graph(restrictSearchWithMatch={"tenant_id": "tenant-b"})])
    clauses = out[1]["$graphLookup"]["restrictSearchWithMatch"]["$and"]
    assert {"tenant_id": T} in clauses  # ANDed, so tenant-b can never match


def test_graph_lookup_of_unlisted_collection_is_rejected():
    with pytest.raises(PipelineGuardError, match="allowed foreign collections"):
        guard([graph(**{"from": "users"})])


def test_graph_lookup_restriction_must_be_a_mapping():
    with pytest.raises(PipelineGuardError, match="must be a mapping"):
        guard([graph(restrictSearchWithMatch="tenant_id")])


def test_graph_lookup_rejects_unknown_options():
    with pytest.raises(PipelineGuardError, match="unsupported"):
        guard([graph(sneaky=1)])


# -- $unionWith -------------------------------------------------------------


def test_union_with_string_form_is_rewritten_and_scoped():
    out = guard([{"$unionWith": "chunks"}])
    assert out[1] == {"$unionWith": {"coll": "chunks", "pipeline": [TENANT_MATCH]}}


def test_union_with_document_form_gets_the_match_before_callers_stages():
    caller = {"$match": {"kind": "x"}}
    out = guard([{"$unionWith": {"coll": "chunks", "pipeline": [caller]}}])
    assert out[1]["$unionWith"]["pipeline"] == [TENANT_MATCH, caller]


def test_union_with_without_a_collection_is_rejected():
    with pytest.raises(PipelineGuardError, match="requires a collection name"):
        guard([{"$unionWith": {"pipeline": [{"$limit": 1}]}}])


def test_union_with_of_unlisted_collection_is_rejected():
    with pytest.raises(PipelineGuardError, match="allowed foreign collections"):
        guard([{"$unionWith": "users"}])


def test_union_with_rejects_unknown_options():
    with pytest.raises(PipelineGuardError, match="unsupported"):
        guard([{"$unionWith": {"coll": "chunks", "sneaky": 1}}])


def test_union_with_rejects_wrong_body_types():
    with pytest.raises(PipelineGuardError, match="collection name or a mapping"):
        guard([{"$unionWith": 5}])


def test_nested_union_with_is_scoped():
    inner = {"$unionWith": "nodes"}
    out = guard([{"$unionWith": {"coll": "chunks", "pipeline": [inner]}}])
    foreign = out[1]["$unionWith"]["pipeline"]
    assert foreign[0] == TENANT_MATCH
    assert foreign[1]["$unionWith"]["pipeline"][0] == TENANT_MATCH


# -- $facet -----------------------------------------------------------------


def test_facet_subpipelines_are_not_given_an_extra_root_match():
    sub = [{"$match": {"k": 1}}]
    out = guard([{"$facet": {"f": sub}}])
    assert out[1]["$facet"]["f"] == sub


def test_write_stage_inside_facet_is_rejected():
    with pytest.raises(PipelineGuardError, match="never allowed"):
        guard([{"$facet": {"f": [{"$merge": {"into": "x"}}]}}])


def test_empty_or_malformed_facet_is_rejected():
    for body in ({}, [], "x"):
        with pytest.raises(PipelineGuardError, match="named sub-pipeline"):
            guard([{"$facet": body}])


# -- limits -----------------------------------------------------------------


def test_deep_nesting_is_rejected_with_a_clean_error_not_a_recursion_error():
    pipeline = [{"$limit": 1}]
    for _ in range(MAX_DEPTH + 4):
        pipeline = [{"$facet": {"f": pipeline}}]
    with pytest.raises(PipelineGuardError, match="too deeply"):
        guard(pipeline)


def test_stage_budget_is_enforced():
    guard([{"$limit": 1}] * MAX_TOTAL_STAGES)  # exactly at the limit is fine
    with pytest.raises(PipelineGuardError, match="too many"):
        guard([{"$limit": 1}] * (MAX_TOTAL_STAGES + 1))


# -- super admin (tenant_id=None) -------------------------------------------


def test_super_admin_gets_no_root_match_but_keeps_every_ban():
    assert guard([{"$limit": 1}], tenant_id=None) == [{"$limit": 1}]
    with pytest.raises(PipelineGuardError, match="not allowed"):
        guard([{"$match": {"$where": "1"}}], tenant_id=None)


def test_super_admin_still_needs_the_allow_list_for_foreign_reads():
    with pytest.raises(PipelineGuardError, match="allowed foreign collections"):
        guard([lookup(frm="users")], tenant_id=None)


def test_super_admin_lookup_is_not_tenant_filtered():
    out = guard([lookup()], tenant_id=None)
    assert out[0]["$lookup"]["pipeline"][0].get("$match", {}).get("tenant_id") is None


# -- argument validation ----------------------------------------------------


@pytest.mark.parametrize("bad", ["", 123, b"tenant"])
def test_tenant_id_must_be_a_non_empty_string(bad):
    with pytest.raises(PipelineGuardError, match="non-empty string"):
        guard([{"$limit": 1}], tenant_id=bad)


def test_allow_list_must_not_be_a_bare_string():
    with pytest.raises(PipelineGuardError, match="not a string"):
        guard([lookup()], allow="chunks")
