"""Offline tests for how API timestamps reach the browser.

Storage hands back naive datetimes in UTC (SQLite/MySQL drop the offset, pymongo
always returns naive), and a bare ``2026-10-20T10:00:00`` string is a *local*
time under ECMAScript. Every response datetime must therefore carry an explicit
offset, or the UI renders each instant shifted by the viewer's timezone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Union, get_args, get_origin

import pytest
from pydantic import BaseModel, PlainSerializer

from app.api import schemas
from app.api.schemas import DocumentOut, JobOut


def _naive_job(created_at: datetime) -> JobOut:
    return JobOut(
        id="job-1", workspace_id="ws-1", kind="ingest", status="queued",
        progress=0.0, error="", created_at=created_at,
    )


def test_naive_utc_datetime_is_serialized_with_an_explicit_offset():
    dumped = _naive_job(datetime(2026, 10, 20, 10, 0, 0)).model_dump_json()
    assert '"created_at":"2026-10-20T10:00:00Z"' in dumped


def test_aware_datetime_is_normalized_to_utc_not_printed_local():
    tokyo = timezone(timedelta(hours=9))
    same_instant = datetime(2026, 10, 20, 19, 0, 0, tzinfo=tokyo)
    dumped = _naive_job(same_instant).model_dump_json()
    assert '"created_at":"2026-10-20T10:00:00Z"' in dumped


def test_nullable_timestamps_stay_nullable():
    dumped = _naive_job(datetime(2026, 10, 20, 10, 0, 0)).model_dump_json()
    assert '"started_at":null' in dumped
    assert '"finished_at":null' in dumped


def test_offset_survives_a_json_round_trip():
    parsed = JobOut.model_validate_json(
        _naive_job(datetime(2026, 10, 20, 10, 0, 0)).model_dump_json()
    )
    assert parsed.created_at.tzinfo is not None
    assert parsed.created_at == datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)


def test_internal_python_dumps_keep_datetime_objects():
    """Only JSON output changes shape; ``model_dump()`` callers are untouched."""
    value = _naive_job(datetime(2026, 10, 20, 10, 0, 0)).model_dump()["created_at"]
    assert isinstance(value, datetime)


def _leaves(annotation):
    """Flatten Optional/Union wrappers into their leaf annotations.

    Only unions are expanded: ``Annotated`` also has ``get_args``, and digging
    into it would strip the very metadata this test looks for.
    """
    if get_origin(annotation) is Union:
        out = []
        for arg in get_args(annotation):
            out.extend(_leaves(arg))
        return out
    return [annotation]


def _unserialized_datetime_fields(model) -> list[str]:
    """Fields whose datetime values would serialize without a UTC offset.

    Pydantic hoists ``Annotated`` metadata onto ``FieldInfo.metadata`` when the
    field is a bare datetime, but leaves it on the member when the field is
    ``Optional[...]`` — both shapes have to be checked.
    """
    bad = []
    for name, info in model.model_fields.items():
        for leaf in _leaves(info.annotation):
            base = get_args(leaf)[0] if getattr(leaf, "__metadata__", None) else leaf
            if base is not datetime:
                continue
            metadata = tuple(getattr(leaf, "__metadata__", ())) + tuple(info.metadata)
            if not any(isinstance(m, PlainSerializer) for m in metadata):
                bad.append(name)
    return bad


@pytest.mark.parametrize(
    "model",
    sorted(
        (
            obj
            for name, obj in vars(schemas).items()
            if isinstance(obj, type) and issubclass(obj, BaseModel)
            and name.endswith("Out")
        ),
        key=lambda m: m.__name__,
    ),
    ids=lambda m: m.__name__,
)
def test_every_response_timestamp_carries_the_utc_serializer(model):
    bad = _unserialized_datetime_fields(model)
    assert bad == [], f"{model.__name__}.{bad} serialize as local time"


def test_document_out_timestamps_are_utc_qualified():
    doc = DocumentOut(
        id="doc-1", workspace_id="ws-1", title="AR 2024", filename="a.pdf",
        doc_type="annual_report", company="ACME", ticker="1000", year=2024,
        quarter="", section="", status="active", parse_meta={},
        created_at=datetime(2026, 10, 20, 10, 0, 0),
    )
    assert '"created_at":"2026-10-20T10:00:00Z"' in doc.model_dump_json()
