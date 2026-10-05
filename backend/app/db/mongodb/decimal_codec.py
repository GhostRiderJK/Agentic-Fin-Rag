"""Exact decimal handling for MongoDB.

Money and ratios are ``decimal.Decimal`` in Python and ``Decimal128`` in
MongoDB, never ``float``. A float has already lost precision before it
reaches the database (``0.1`` is not one tenth), and a financial platform
must not quietly store or compare such values.

Rules enforced here, at the repository boundary:

* Fields a repository declares in ``decimal_fields`` accept ``Decimal``,
  ``int`` and numeric ``str``. ``float`` and ``bool`` are rejected, as are
  ``NaN`` / ``Infinity`` and values Decimal128 cannot hold exactly (more than
  34 significant digits). Nothing is silently rounded.
* Any other ``Decimal`` found in a document, filter, update or pipeline is
  converted to ``Decimal128`` too, so callers never mix types.
* Reads convert every ``Decimal128`` back to ``Decimal``.

Only top-level declared fields are checked strictly in filters; a float buried
inside ``$and`` / ``$or`` is passed through unchanged (it simply will not
match an exact decimal).
"""

from __future__ import annotations

import decimal
from decimal import Decimal
from typing import Any, Iterable, Mapping, Optional

from bson.decimal128 import Decimal128

__all__ = [
    "DecimalFieldError",
    "decode_document",
    "decode_value",
    "encode_document",
    "encode_filter",
    "encode_query",
    "encode_update",
    "to_decimal128",
]


class DecimalFieldError(ValueError):
    """Raised when a value cannot be stored or compared as an exact decimal."""


_UPDATE_OPERATORS = ("$set", "$setOnInsert", "$inc", "$mul", "$min", "$max")
_COMPARISON_OPERATORS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"})
_LIST_OPERATORS = frozenset({"$in", "$nin"})


def _label(field: str) -> str:
    return f"field {field!r}: " if field else ""


def to_decimal128(value: Any, *, field: str = "") -> Decimal128:
    """Strictly convert ``value`` to ``Decimal128`` or raise DecimalFieldError."""
    if isinstance(value, bool):
        raise DecimalFieldError(f"{_label(field)}bool is not a decimal value")
    if isinstance(value, float):
        raise DecimalFieldError(
            f"{_label(field)}float is not allowed for exact decimals; "
            "pass Decimal, int or str"
        )
    if isinstance(value, Decimal128):
        number = value.to_decimal()
    elif isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except decimal.InvalidOperation:
            raise DecimalFieldError(
                f"{_label(field)}{value!r} is not a valid decimal number"
            ) from None
    else:
        raise DecimalFieldError(
            f"{_label(field)}unsupported type {type(value).__name__} for a decimal value"
        )

    if not number.is_finite():
        raise DecimalFieldError(f"{_label(field)}NaN and Infinity are not allowed")
    if isinstance(value, Decimal128):
        return value
    try:
        encoded = Decimal128(number)
    except (decimal.InvalidOperation, decimal.Overflow, decimal.Inexact):
        raise DecimalFieldError(
            f"{_label(field)}value cannot be represented exactly as Decimal128"
        ) from None
    if encoded.to_decimal() != number:
        raise DecimalFieldError(
            f"{_label(field)}value exceeds Decimal128 precision (34 significant digits)"
        )
    return encoded


# -- writes -----------------------------------------------------------------


def encode_query(obj: Any) -> Any:
    """Convert every ``Decimal`` inside ``obj`` to ``Decimal128`` (generic)."""
    if isinstance(obj, Decimal):
        return to_decimal128(obj)
    if isinstance(obj, Mapping):
        return {key: encode_query(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_query(value) for value in obj]
    return obj


def encode_document(
    doc: Mapping[str, Any], decimal_fields: Iterable[str] = ()
) -> dict:
    """Return a copy of ``doc`` ready to store.

    Declared decimal fields are converted strictly; other ``Decimal`` values
    are converted generically; everything else is left alone.
    """
    declared = frozenset(decimal_fields)
    out: dict = {}
    for key, value in doc.items():
        if key in declared and value is not None:
            out[key] = to_decimal128(value, field=key)
        else:
            out[key] = encode_query(value)
    return out


def encode_update(
    update: Mapping[str, Any], decimal_fields: Iterable[str] = ()
) -> dict:
    """Encode an operator-form update document (``{"$set": {...}}``)."""
    out: dict = {}
    for operator, body in update.items():
        if operator in _UPDATE_OPERATORS and isinstance(body, Mapping):
            out[operator] = encode_document(body, decimal_fields)
        else:
            out[operator] = encode_query(body)
    return out


def _encode_condition(condition: Any, field: str) -> Any:
    if isinstance(condition, Mapping) and any(
        isinstance(key, str) and key.startswith("$") for key in condition
    ):
        out: dict = {}
        for operator, value in condition.items():
            if operator in _COMPARISON_OPERATORS:
                out[operator] = (
                    None if value is None else to_decimal128(value, field=field)
                )
            elif operator in _LIST_OPERATORS:
                out[operator] = [to_decimal128(v, field=field) for v in value]
            else:
                out[operator] = encode_query(value)
        return out
    return None if condition is None else to_decimal128(condition, field=field)


def encode_filter(
    query: Mapping[str, Any], decimal_fields: Iterable[str] = ()
) -> dict:
    """Encode a query filter; declared decimal fields are checked strictly."""
    declared = frozenset(decimal_fields)
    out: dict = {}
    for key, condition in query.items():
        if key in declared:
            out[key] = _encode_condition(condition, key)
        else:
            out[key] = encode_query(condition)
    return out


# -- reads ------------------------------------------------------------------


def decode_value(value: Any) -> Any:
    """Convert every ``Decimal128`` inside ``value`` back to ``Decimal``."""
    if isinstance(value, Decimal128):
        return value.to_decimal()
    if isinstance(value, Mapping):
        return {key: decode_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [decode_value(inner) for inner in value]
    return value


def decode_document(doc: Optional[Mapping[str, Any]]) -> Optional[dict]:
    """Decode a stored document; ``None`` passes through."""
    return None if doc is None else decode_value(doc)
