"""Collection names and index plan.

Every tenant-owned collection gets a compound index whose **first key is
``tenant_id``**. That ordering matters: tenant-scoped queries can use the
index prefix, so isolation is not just correct but cheap.

Tenant-independent collections (``tenants``) are indexed on their own id.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from pymongo import ASCENDING, DESCENDING, IndexModel

# -- collection names ---------------------------------------------------------

TENANTS = "tenants"
USERS = "users"
WORKSPACES = "workspaces"
DOCUMENTS = "documents"
CHUNKS = "chunks"
STRUCTURED_FINANCIALS = "structured_financials"
CONVERSATIONS = "conversations"
MESSAGES = "messages"
JOBS = "jobs"
EVAL_RUNS = "eval_runs"
EVAL_ITEMS = "eval_items"
AUDIT_LOG = "audit_log"
FEEDBACK = "feedback"

#: Collections that carry ``tenant_id`` and are therefore repository-guarded.
TENANT_OWNED: Tuple[str, ...] = (
    USERS,
    WORKSPACES,
    DOCUMENTS,
    CHUNKS,
    STRUCTURED_FINANCIALS,
    CONVERSATIONS,
    MESSAGES,
    JOBS,
    EVAL_RUNS,
    EVAL_ITEMS,
    AUDIT_LOG,
    FEEDBACK,
)

#: All collections the bootstrap script creates.
ALL_COLLECTIONS: Tuple[str, ...] = (TENANTS,) + TENANT_OWNED


# -- index plan ---------------------------------------------------------------


def index_plan() -> Dict[str, List[IndexModel]]:
    """Return ``{collection: [IndexModel, ...]}`` for bootstrap."""
    t = ASCENDING
    d = DESCENDING
    plan: Dict[str, List[IndexModel]] = {
        TENANTS: [
            IndexModel([("id", t)], name="uq_tenant_id", unique=True),
            IndexModel([("name", t)], name="ix_tenant_name"),
        ],
        USERS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_user_id", unique=True),
            # Emails are unique *per tenant*, not globally.
            IndexModel(
                [("tenant_id", t), ("email", t)],
                name="uq_user_tenant_email",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", t), ("api_key", t)],
                name="uq_user_tenant_api_key",
                unique=True,
                partialFilterExpression={"api_key": {"$type": "string"}},
            ),
        ],
        WORKSPACES: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_workspace_id", unique=True),
            IndexModel([("tenant_id", t), ("created_at", d)], name="ix_ws_tenant_created"),
        ],
        DOCUMENTS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_document_id", unique=True),
            IndexModel(
                [("tenant_id", t), ("workspace_id", t), ("status", t)],
                name="ix_doc_scope_status",
            ),
            # Metadata prefilter used by hybrid retrieval.
            IndexModel(
                [("tenant_id", t), ("company", t), ("year", d), ("quarter", t)],
                name="ix_doc_company_period",
            ),
            IndexModel(
                [("tenant_id", t), ("doc_type", t), ("year", d)],
                name="ix_doc_type_year",
            ),
        ],
        CHUNKS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_chunk_id", unique=True),
            IndexModel([("tenant_id", t), ("document_id", t)], name="ix_chunk_document"),
            IndexModel([("tenant_id", t), ("parent_id", t)], name="ix_chunk_parent"),
        ],
        STRUCTURED_FINANCIALS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_financial_id", unique=True),
            IndexModel(
                [
                    ("tenant_id", t),
                    ("workspace_id", t),
                    ("company", t),
                    ("metric", t),
                    ("fiscal_year", d),
                ],
                name="ix_fin_metric_lookup",
            ),
            IndexModel(
                [("tenant_id", t), ("company", t), ("fiscal_year", d), ("quarter", t)],
                name="ix_fin_company_period",
            ),
        ],
        CONVERSATIONS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_conversation_id", unique=True),
            IndexModel(
                [("tenant_id", t), ("workspace_id", t), ("updated_at", d)],
                name="ix_conv_scope_updated",
            ),
            IndexModel([("tenant_id", t), ("user_id", t)], name="ix_conv_user"),
        ],
        MESSAGES: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_message_id", unique=True),
            IndexModel(
                [("tenant_id", t), ("conversation_id", t), ("created_at", t)],
                name="ix_msg_conversation",
            ),
        ],
        JOBS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_job_id", unique=True),
            IndexModel(
                [("tenant_id", t), ("status", t), ("created_at", d)],
                name="ix_job_status",
            ),
            # Expire finished jobs after a retention window.
            IndexModel(
                [("finished_at", t)],
                name="ttl_job_finished",
                expireAfterSeconds=60 * 60 * 24 * 30,
                sparse=True,
            ),
        ],
        EVAL_RUNS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_eval_run_id", unique=True),
            IndexModel([("tenant_id", t), ("created_at", d)], name="ix_eval_run_created"),
        ],
        EVAL_ITEMS: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_eval_item_id", unique=True),
            IndexModel([("tenant_id", t), ("run_id", t)], name="ix_eval_item_run"),
        ],
        AUDIT_LOG: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_audit_id", unique=True),
            IndexModel(
                [("tenant_id", t), ("created_at", d)], name="ix_audit_tenant_created"
            ),
            IndexModel([("tenant_id", t), ("action", t)], name="ix_audit_action"),
        ],
        FEEDBACK: [
            IndexModel([("tenant_id", t), ("id", t)], name="uq_feedback_id", unique=True),
            IndexModel(
                [("tenant_id", t), ("message_id", t)], name="ix_feedback_message"
            ),
        ],
    }
    return plan
