"""Unit tests for app.db.mongodb.decimal_codec (no database needed)."""

from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("bson", reason="pymongo/bson not installed")

from bson.decimal128 import Decimal128  # noqa: E402

from app.db.mongodb.decimal_codec import (  # noqa: E402
    DecimalFieldError,
    decode_document,
    encode_document,
    encode_filter,
    encode_query,
    encode_update,
    to_decimal128,
)

FIELDS = frozenset({"value"})
LONG = Decimal("123.456789012345678901")  # 21 significant digits: exact in Decimal128


# -- to_decimal128 ----------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("1500.25"), Decimal("1500.25")),
        (42, Decimal("42")),
        ("1500.25", Decimal("1500.25")),
        ("  7.5  ", Decimal("7.5")),
        (LONG, LONG),
        (Decimal("-0.000001"), Decimal("-0.000001")),
    ],
)
def test_accepts_exact_values(value, expected):
    encoded = to_decimal128(value)
    assert isinstance(encoded, Decimal128)
    assert encoded.to_decimal() == expected


def test_preserves_scale():
    assert str(to_decimal128(Decimal("1.50")).to_decimal()) == "1.50"


@pytest.mark.parametrize("value", [1.5, 0.1, float("nan"), float("inf")])
def test_floats_are_rejected(value):
    with pytest.raises(DecimalFieldError, match="float"):
        to_decimal128(value, field="value")


@pytest.mark.parametrize("value", [True, False])
def test_bools_are_rejected(value):
    with pytest.raises(DecimalFieldError, match="bool"):
        to_decimal128(value)


@pytest.mark.parametrize(
    "value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), "NaN", "Infinity"]
)
def test_non_finite_values_are_rejected(value):
    with pytest.raises(DecimalFieldError, match="NaN and Infinity"):
        to_decimal128(value)


@pytest.mark.parametrize("value", ["abc", "", "1,000.00", "12 34"])
def test_garbage_strings_are_rejected(value):
    with pytest.raises(DecimalFieldError, match="not a valid decimal"):
        to_decimal128(value)


def test_unsupported_types_are_rejected():
    with pytest.raises(DecimalFieldError, match="unsupported type"):
        to_decimal128([1])


def test_more_than_34_digits_is_rejected_not_rounded():
    too_long = Decimal("1." + "0" * 40 + "1")
    with pytest.raises(DecimalFieldError):
        to_decimal128(too_long)


def test_decimal128_passes_through():
    original = Decimal128(Decimal("9.99"))
    assert to_decimal128(original) is original


def test_error_message_names_the_field():
    with pytest.raises(DecimalFieldError, match="field 'value'"):
        to_decimal128(1.5, field="value")


# -- encode_query (generic) -------------------------------------------------


def test_encode_query_converts_nested_decimals_only():
    out = encode_query(
        {"a": Decimal("1.5"), "b": [Decimal("2"), {"c": Decimal("3")}], "d": 1.5, "e": "x"}
    )
    assert isinstance(out["a"], Decimal128)
    assert isinstance(out["b"][0], Decimal128)
    assert isinstance(out["b"][1]["c"], Decimal128)
    assert out["d"] == 1.5 and out["e"] == "x"


def test_encode_query_rejects_non_finite_decimals():
    with pytest.raises(DecimalFieldError):
        encode_query({"a": [Decimal("NaN")]})


# -- encode_document --------------------------------------------------------


def test_declared_fields_are_strict_and_others_are_untouched():
    out = encode_document({"value": "12.5", "name": "x", "ratio": 0.25}, FIELDS)
    assert isinstance(out["value"], Decimal128)
    assert out["name"] == "x"
    assert out["ratio"] == 0.25  # undeclared float is left alone


def test_declared_field_rejects_float():
    with pytest.raises(DecimalFieldError, match="float"):
        encode_document({"value": 120.0}, FIELDS)


def test_declared_field_allows_none():
    assert encode_document({"value": None}, FIELDS) == {"value": None}


def test_undeclared_decimals_are_still_converted():
    out = encode_document({"other": Decimal("1.1")}, FIELDS)
    assert isinstance(out["other"], Decimal128)


def test_encode_document_does_not_mutate_input():
    doc = {"value": "1.5"}
    encode_document(doc, FIELDS)
    assert doc == {"value": "1.5"}


# -- encode_update ----------------------------------------------------------


def test_update_set_and_inc_are_strict_for_declared_fields():
    out = encode_update({"$set": {"value": "2.5"}, "$inc": {"value": 1}}, FIELDS)
    assert isinstance(out["$set"]["value"], Decimal128)
    assert isinstance(out["$inc"]["value"], Decimal128)


@pytest.mark.parametrize("operator", ["$set", "$inc", "$mul", "$min", "$max", "$setOnInsert"])
def test_update_rejects_floats_for_declared_fields(operator):
    with pytest.raises(DecimalFieldError, match="float"):
        encode_update({operator: {"value": 1.5}}, FIELDS)


def test_update_leaves_other_operators_alone():
    out = encode_update({"$unset": {"value": ""}}, FIELDS)
    assert out == {"$unset": {"value": ""}}


# -- encode_filter ----------------------------------------------------------


def test_filter_converts_declared_equality_and_comparisons():
    out = encode_filter(
        {"value": {"$gt": "100", "$lte": Decimal("200")}, "company": "ACME"}, FIELDS
    )
    assert isinstance(out["value"]["$gt"], Decimal128)
    assert isinstance(out["value"]["$lte"], Decimal128)
    assert out["company"] == "ACME"
    assert isinstance(encode_filter({"value": Decimal("1")}, FIELDS)["value"], Decimal128)


def test_filter_converts_in_lists():
    out = encode_filter({"value": {"$in": [1, "2", Decimal("3")]}}, FIELDS)
    assert all(isinstance(v, Decimal128) for v in out["value"]["$in"])


@pytest.mark.parametrize(
    "condition", [1.5, {"$gt": 1.5}, {"$in": [1, 2.5]}, {"$ne": 0.1}]
)
def test_filter_rejects_floats_on_declared_fields(condition):
    with pytest.raises(DecimalFieldError, match="float"):
        encode_filter({"value": condition}, FIELDS)


def test_filter_allows_null_comparison_on_declared_fields():
    assert encode_filter({"value": None}, FIELDS) == {"value": None}
    assert encode_filter({"value": {"$eq": None}}, FIELDS) == {"value": {"$eq": None}}


def test_filter_leaves_undeclared_floats_alone():
    assert encode_filter({"ratio": {"$gt": 0.5}}, FIELDS) == {"ratio": {"$gt": 0.5}}


# -- decode -----------------------------------------------------------------


def test_decode_converts_nested_decimal128_back_to_decimal():
    stored = {
        "_id": "keep",
        "value": Decimal128(LONG),
        "rows": [{"v": Decimal128(Decimal("2.50"))}],
        "name": "x",
    }
    out = decode_document(stored)
    assert out["value"] == LONG and isinstance(out["value"], Decimal)
    assert out["rows"][0]["v"] == Decimal("2.50")
    assert str(out["rows"][0]["v"]) == "2.50"
    assert out["_id"] == "keep" and out["name"] == "x"


def test_decode_passes_none_through():
    assert decode_document(None) is None


def test_encode_then_decode_is_lossless():
    doc = {"value": LONG, "nested": {"v": Decimal("0.1")}}
    encoded = encode_document(doc, FIELDS)
    assert decode_document(encoded) == doc
