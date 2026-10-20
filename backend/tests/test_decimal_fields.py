"""Money and ratio fields are exact decimals in the *production* repositories.

test_mongo_hardening.py proves the codec works using a test-local repository.
This file proves the real repositories actually declare their money columns,
because the codec only enforces what ``decimal_fields`` names — an undeclared
column is stored as a float with no error at all.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

pytest.importorskip("bson", reason="pymongo/bson not installed")

from bson.decimal128 import Decimal128  # noqa: E402

from app.db.mongodb import collections as C  # noqa: E402
from app.db.mongodb.decimal_codec import DecimalFieldError  # noqa: E402
from app.db.mongodb.repositories import (  # noqa: E402
    JobRepository,
    StructuredFinancialRepository,
)

if not os.path.exists(os.path.join(os.path.dirname(__file__), '__init__.py')):
    from mongo_support import ctx_a, db, run  # noqa: F401
else:
    from tests.mongo_support import ctx_a, db, run  # noqa: F401


def _raw(run, collection_name: str, doc_id: str, tenant_id: str):
    """Read the stored document without the repository, to see the BSON types."""
    from app.db.mongodb.client import get_database

    async def _go():
        return await get_database()[collection_name].find_one(
            {"id": doc_id, "tenant_id": tenant_id}
        )

    return run(_go())


def _financial(**overrides):
    doc = {
        "company": "ACME",
        "metric": "revenue",
        "fiscal_year": 2025,
        "quarter": "Q1",
        "value": Decimal("1234.56"),
        "confidence": Decimal("0.85"),
    }
    doc.update(overrides)
    return doc


# -- structured_financials ----------------------------------------------------


def test_financials_declare_value_and_confidence_as_decimals():
    assert StructuredFinancialRepository.decimal_fields == frozenset(
        {"value", "confidence"}
    )


def test_money_round_trips_with_its_scale_intact(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    inserted = run(repo.insert_one(ctx_a, _financial(value=Decimal("1234.5600"))))
    assert inserted["value"] == Decimal("1234.5600")
    # Equality alone would pass after silent rounding; the text proves no
    # normalisation happened on the way in.
    assert str(inserted["value"]) == "1234.5600"
    assert str(inserted["confidence"]) == "0.85"

    stored = _raw(run, C.STRUCTURED_FINANCIALS, inserted["id"], ctx_a.tenant_id)
    assert isinstance(stored["value"], Decimal128)
    assert isinstance(stored["confidence"], Decimal128)

    read_back = run(repo.get_by_id(ctx_a, inserted["id"]))
    assert str(read_back["value"]) == "1234.5600"


def test_a_float_money_value_is_rejected_and_nothing_is_stored(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    with pytest.raises(DecimalFieldError, match="float is not allowed"):
        run(repo.insert_one(ctx_a, _financial(value=1234.56)))
    assert run(repo.count(ctx_a, {})) == 0


def test_a_float_confidence_is_rejected_too(db, run, ctx_a):
    """The ratio columns are the ones people forget, because 'it's only a
    confidence score'."""
    repo = StructuredFinancialRepository()
    with pytest.raises(DecimalFieldError, match="confidence"):
        run(repo.insert_one(ctx_a, _financial(confidence=0.85)))


def test_bool_and_nan_are_not_money(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    with pytest.raises(DecimalFieldError, match="bool"):
        run(repo.insert_one(ctx_a, _financial(value=True)))
    with pytest.raises(DecimalFieldError, match="NaN"):
        run(repo.insert_one(ctx_a, _financial(value=Decimal("NaN"))))


def test_decimal128_precision_limit_is_refused_not_rounded(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    too_precise = Decimal("1." + "2" * 40)
    with pytest.raises(DecimalFieldError, match="exactly|precision"):
        run(repo.insert_one(ctx_a, _financial(value=too_precise)))
    assert run(repo.count(ctx_a, {})) == 0


def test_filtering_money_by_a_float_is_rejected(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    run(repo.insert_one(ctx_a, _financial()))
    with pytest.raises(DecimalFieldError, match="float is not allowed"):
        run(repo.find_many(ctx_a, {"value": 1234.56}))
    exact = run(repo.find_many(ctx_a, {"value": Decimal("1234.56")}))
    assert len(exact) == 1


def test_updating_money_with_a_float_is_rejected(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    doc = run(repo.insert_one(ctx_a, _financial()))
    with pytest.raises(DecimalFieldError, match="float is not allowed"):
        run(repo.update_by_id(ctx_a, doc["id"], {"value": 99.99}))
    assert run(repo.update_by_id(ctx_a, doc["id"],
                                 {"value": Decimal("99.99")})) == 1
    assert str(run(repo.get_by_id(ctx_a, doc["id"]))["value"]) == "99.99"


def test_metric_reads_return_decimals_not_floats(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    run(repo.insert_one(ctx_a, _financial()))
    rows = run(repo.get_metric(ctx_a, company="ACME", metric="revenue",
                               fiscal_year=2025))
    assert len(rows) == 1
    assert isinstance(rows[0]["value"], Decimal)


def test_summing_money_is_exact(db, run, ctx_a):
    """0.1 + 0.2 is the classic float failure; money must not inherit it."""
    repo = StructuredFinancialRepository()
    run(repo.insert_one(ctx_a, _financial(metric="a", value=Decimal("0.1"))))
    run(repo.insert_one(ctx_a, _financial(metric="b", value=Decimal("0.2"))))
    totals = run(repo.aggregate(ctx_a, [
        {"$group": {"_id": None, "total": {"$sum": "$value"}}},
    ]))
    assert str(totals[0]["total"]) == "0.3"


# -- jobs ---------------------------------------------------------------------


def test_jobs_declare_progress_as_a_decimal():
    assert JobRepository.decimal_fields == frozenset({"progress"})


def test_job_progress_round_trips_and_refuses_a_float(db, run, ctx_a):
    repo = JobRepository()
    doc = run(repo.insert_one(ctx_a, {"kind": "ingest", "status": "running",
                                      "progress": Decimal("0.25")}))
    assert str(doc["progress"]) == "0.25"
    stored = _raw(run, C.JOBS, doc["id"], ctx_a.tenant_id)
    assert isinstance(stored["progress"], Decimal128)

    with pytest.raises(DecimalFieldError, match="progress"):
        run(repo.update_by_id(ctx_a, doc["id"], {"progress": 0.5}))
    assert run(repo.update_by_id(ctx_a, doc["id"],
                                 {"progress": Decimal("0.5")})) == 1


# -- coverage -----------------------------------------------------------------


#: Money columns that have already moved to MongoDB, so the SQL layer no
#: longer has a ``Float`` column to compare them against. Item (i) deletes the
#: SQLAlchemy layer and this list with it.
_MIGRATED = {"structured_financials.value", "structured_financials.confidence"}


def test_every_float_column_in_the_legacy_schema_is_declared_or_migrated():
    """Anything the SQL layer still stores as ``Float`` is a money/ratio
    candidate.

    This is the drift guard while both layers exist: adding a Float column
    without deciding its Mongo counterpart's ``decimal_fields`` fails here,
    instead of quietly becoming another 0.30000000000000004. It retires with
    the SQLAlchemy layer.
    """
    import sqlalchemy as sa

    from app.db import models

    float_columns = {
        f"{table.name}.{column.name}"
        for table in models.Base.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, sa.Float)
    }

    declared = {
        f"{repo.collection_name}.{field_name}"
        for repo in (StructuredFinancialRepository(), JobRepository())
        for field_name in repo.decimal_fields
    }
    # The SQL table names and the Mongo collection names coincide for the two
    # money-bearing tables, which is what makes this comparison meaningful.
    undeclared = float_columns - declared
    assert not undeclared, f"undeclared money/ratio columns: {sorted(undeclared)}"
    stale = declared - float_columns - _MIGRATED
    assert not stale, (
        "decimal_fields naming neither a SQL Float column nor a migrated "
        f"collection: {sorted(stale)}"
    )
