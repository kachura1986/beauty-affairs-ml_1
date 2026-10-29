import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from services.agent.tools.truncation import MAX_DEPTH, TRUNCATED_FIELDS_KEY, TRUNCATED_LIST_KEY, truncate_for_model


# ---------------------------------------------------------------------------
# Basic: passthrough
# ---------------------------------------------------------------------------

def test_small_payload_unchanged():
    """Small payloads pass through unchanged."""
    content, truncated = truncate_for_model({"id": 1, "name": "test"}, 1000)
    assert not truncated
    assert json.loads(content) == {"id": 1, "name": "test"}


@pytest.mark.parametrize("value,expected", [
    (None, None),
    (True, True),
    (False, False),
    (42, 42),
    (3.14, 3.14),
    ("hello", "hello"),
    ("привет", "привет"),
    ([], []),
    ({}, {}),
])
def test_scalars_roundtrip(value, expected):
    """Scalars survive a JSON round-trip without truncation."""
    content, truncated = truncate_for_model(value, 1000)
    assert not truncated
    assert json.loads(content) == expected


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

def test_list_truncated_from_end():
    """Lists are truncated from the end and receive a marker."""
    content, truncated = truncate_for_model(list(range(100)), 200)
    assert truncated
    assert len(content) <= 200
    parsed = json.loads(content)
    assert isinstance(parsed, list)
    assert parsed[-1][TRUNCATED_LIST_KEY] is True
    assert parsed[-1]["total"] == 100
    assert parsed[-1]["returned"] == len(parsed) - 1
    assert parsed[0] == 0


def test_list_of_dicts_truncated():
    """Lists of dictionaries are truncated while preserving list type."""
    items = [{"id": i, "name": f"item-{i}"} for i in range(50)]
    content, truncated = truncate_for_model(items, 300)
    assert truncated
    assert len(content) <= 300
    parsed = json.loads(content)
    assert isinstance(parsed, list)
    assert parsed[-1][TRUNCATED_LIST_KEY] is True
    assert parsed[-1]["total"] == 50


def test_empty_list_stays_empty():
    """An empty list is not truncated."""
    content, truncated = truncate_for_model([], 100)
    assert not truncated
    assert json.loads(content) == []


def test_list_stays_list_in_fallback():
    """Regression: the minimal fallback used to return a dict instead of a list.

    The original JSON for range(100) is about 390 characters. The minimal list marker
    [{"_truncated": true, "total": 100}] is about 36 characters, so with a budget of 40 truncation is
    guaranteed and the result must remain a list.
    """
    content, truncated = truncate_for_model(list(range(100)), 40)
    assert truncated
    assert len(content) <= 40
    parsed = json.loads(content)
    assert isinstance(parsed, list)


def test_truncated_total_preserved_on_reduction():
    """When an already truncated list is reduced again, `total` is preserved."""
    content, _ = truncate_for_model(list(range(100)), 200)
    parsed = json.loads(content)
    assert parsed[-1]["total"] == 100

    content2, _ = truncate_for_model(parsed, 100)
    parsed2 = json.loads(content2)
    if parsed2 and isinstance(parsed2[-1], dict) and parsed2[-1].get(TRUNCATED_LIST_KEY):
        assert parsed2[-1]["total"] == 100


# ---------------------------------------------------------------------------
# Dictionaries: nested containers
# ---------------------------------------------------------------------------

def test_dict_with_nested_list_truncated():
    """A dictionary with a nested list is reduced while preserving outer fields."""
    data = {"total": 100, "items": [{"id": i} for i in range(100)]}
    content, truncated = truncate_for_model(data, 400)
    assert truncated
    assert len(content) <= 400
    parsed = json.loads(content)
    assert parsed["total"] == 100
    assert isinstance(parsed["items"], list)
    assert parsed["items"][-1][TRUNCATED_LIST_KEY] is True


def test_dict_with_nested_dict_truncated():
    """A dictionary with a nested dictionary is reduced while preserving metadata fields."""
    data = {
        "meta": {"version": 1, "source": "catalog"},
        "payload": {f"k{i}": "x" * 100 for i in range(50)},
    }
    content, truncated = truncate_for_model(data, 300)
    assert truncated
    assert len(content) <= 300
    parsed = json.loads(content)
    assert parsed["meta"]["version"] == 1


def test_nested_list_preserves_type():
    """Regression: a list field must not be converted into a string."""
    data = {"items": list(range(100))}
    content, truncated = truncate_for_model(data, 100)
    assert truncated
    parsed = json.loads(content)
    assert "items" in parsed
    assert isinstance(parsed["items"], list)


# ---------------------------------------------------------------------------
# Dictionaries: long strings
# ---------------------------------------------------------------------------

def test_dict_with_long_string_truncated():
    """Long string values are shortened while scalar fields survive."""
    data = {"id": 1, "description": "x" * 5000, "price": 99.5}
    content, truncated = truncate_for_model(data, 300)
    assert truncated
    assert len(content) <= 300
    parsed = json.loads(content)
    assert parsed["id"] == 1
    assert parsed["price"] == 99.5
    assert "truncated" in parsed["description"]


# ---------------------------------------------------------------------------
# Dictionaries: field dropping
# ---------------------------------------------------------------------------

def test_dict_fields_dropped():
    """Fields are dropped and recorded in `_truncated_fields` when needed."""
    data = {f"k{i}": "v" * 200 for i in range(20)}
    content, truncated = truncate_for_model(data, 200)
    assert truncated
    assert len(content) <= 200
    parsed = json.loads(content)
    assert TRUNCATED_FIELDS_KEY in parsed
    assert len(parsed[TRUNCATED_FIELDS_KEY]) > 0


def test_dict_all_fields_dropped_when_tiny_budget():
    """All fields can be dropped for a very small budget, but the result remains valid JSON."""
    data = {"a": "x" * 1000, "b": "y" * 1000}
    content, truncated = truncate_for_model(data, 60)
    assert truncated
    assert len(content) <= 60
    json.loads(content)


# ---------------------------------------------------------------------------
# Collision of `_truncated_fields` with input data
# ---------------------------------------------------------------------------

def test_marker_key_collision_with_non_list_value():
    """A non-list `_truncated_fields` value in input data is not overwritten by the marker."""
    data: dict[str, Any] = {f"k{i}": "v" * 200 for i in range(20)}
    data[TRUNCATED_FIELDS_KEY] = "original"

    content, truncated = truncate_for_model(data, 200)
    assert truncated
    assert len(content) <= 200
    parsed = json.loads(content)
    if TRUNCATED_FIELDS_KEY in parsed:
        assert parsed[TRUNCATED_FIELDS_KEY] == "original"


def test_marker_key_append_to_existing_list():
    """An existing list `_truncated_fields` value is preserved when fields are dropped."""
    data: dict[str, Any] = {f"k{i}": "v" * 200 for i in range(20)}
    data[TRUNCATED_FIELDS_KEY] = ["prior"]

    content, truncated = truncate_for_model(data, 200)
    assert truncated
    parsed = json.loads(content)
    if TRUNCATED_FIELDS_KEY in parsed and isinstance(parsed[TRUNCATED_FIELDS_KEY], list):
        assert "prior" in parsed[TRUNCATED_FIELDS_KEY]


# ---------------------------------------------------------------------------
# Regression: previously caused an infinite loop
# ---------------------------------------------------------------------------

def test_many_small_fields_do_not_hang():
    """Regression: many small fields must not cause an infinite loop."""
    data = {f"k{i}": "v" * 50 for i in range(30)}
    content, truncated = truncate_for_model(data, 100)
    assert truncated
    assert len(content) <= 100
    json.loads(content)


def test_single_field_string_does_not_hang():
    """Regression: a single long string field must not cause an infinite loop."""
    data = {"only": "z" * 5000}
    content, truncated = truncate_for_model(data, 80)
    assert truncated
    assert len(content) <= 80
    parsed = json.loads(content)
    assert "only" in parsed or TRUNCATED_FIELDS_KEY in parsed


# ---------------------------------------------------------------------------
# Compression of an overgrown `_truncated_fields`
# ---------------------------------------------------------------------------

def test_truncated_fields_compressed():
    """An overgrown `_truncated_fields` marker is compressed to fit the budget."""
    data = {f"k{i}": "v" * 30 for i in range(50)}
    content, truncated = truncate_for_model(data, 100)
    assert truncated
    assert len(content) <= 100
    parsed = json.loads(content)
    if TRUNCATED_FIELDS_KEY in parsed:
        assert isinstance(parsed[TRUNCATED_FIELDS_KEY], list)


# ---------------------------------------------------------------------------
# Pydantic / dataclass / misc
# ---------------------------------------------------------------------------

class _Product(BaseModel):
    """Pydantic model used to verify serialization."""

    id: int
    name: str
    price: float


def test_pydantic_model():
    """Pydantic models are serialized to JSON objects."""
    content, truncated = truncate_for_model(_Product(id=1, name="t", price=9.5), 1000)
    assert not truncated
    assert json.loads(content) == {"id": 1, "name": "t", "price": 9.5}


@dataclass
class _Point:
    """Dataclass used to verify serialization."""

    x: int
    y: int


def test_dataclass():
    """Dataclasses are serialized to JSON objects."""
    content, _ = truncate_for_model(_Point(1, 2), 1000)
    assert json.loads(content) == {"x": 1, "y": 2}


def test_decimal_is_number():
    """Decimal values are serialized as JSON numbers."""
    content, _ = truncate_for_model({"price": Decimal("19.99")}, 1000)
    parsed = json.loads(content)
    assert isinstance(parsed["price"], (int, float))
    assert parsed["price"] == pytest.approx(19.99, abs=0.01)


# ---------------------------------------------------------------------------
# Cycles / depth / special values
# ---------------------------------------------------------------------------

def test_cycle_handling():
    """Dictionary cycles are replaced with a cycle marker."""
    a: dict = {}
    a["self"] = a
    content, _ = truncate_for_model(a, 1000)
    assert json.loads(content)["self"] == "<cycle>"


def test_cycle_handling_with_list():
    """List cycles are replaced with a cycle marker."""
    a: list = [1, 2]
    a.append(a)
    content, _ = truncate_for_model(a, 1000)
    parsed = json.loads(content)
    assert parsed[2] == "<cycle>"


def test_nan_infinity():
    """NaN and infinities are serialized as strings."""
    content, _ = truncate_for_model(
        {"nan": float("nan"), "inf": float("inf"), "neg": float("-inf")}, 1000
    )
    parsed = json.loads(content)
    assert parsed == {"nan": "nan", "inf": "inf", "neg": "-inf"}


def test_deeply_nested_depth_marker():
    """Excessively deep nesting is replaced with a depth marker."""
    data: dict = {}
    cur = data
    for _ in range(MAX_DEPTH + 10):
        cur["child"] = {}
        cur = cur["child"]
    content, _ = truncate_for_model(data, 5000)
    assert "<max-depth>" in content


def test_bytes_set_tuple():
    """Bytes, sets, and tuples are normalized to JSON-compatible values."""
    assert json.loads(truncate_for_model(b"hi", 1000)[0]) == "hi"
    assert json.loads(truncate_for_model({3, 1, 2}, 1000)[0]) == [1, 2, 3]
    assert json.loads(truncate_for_model((1, 2, 3), 1000)[0]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Scalars too large for the budget
# ---------------------------------------------------------------------------

def test_huge_int_returns_valid_fallback():
    """A huge integer returns a valid fallback when it cannot fit."""
    big = 10 ** 2000
    content, truncated = truncate_for_model({"n": big}, 20)
    assert truncated
    assert len(content) <= 20
    json.loads(content)


def test_huge_bool_string_cannot_be_broken():
    """A boolean cannot be partially truncated; it stays valid."""
    content, _ = truncate_for_model(True, 100)
    assert json.loads(content) is True


# ---------------------------------------------------------------------------
# JSON validity across a reasonable range of budgets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_chars", [50, 100, 200, 500, 1000, 5000])
def test_always_valid_json_for_reasonable_limits(max_chars):
    """The result is valid JSON and within budget for reasonable limits."""
    data = {
        "items": [{"id": i, "desc": "x" * 100} for i in range(100)],
        "note": "y" * 500,
    }
    content, _ = truncate_for_model(data, max_chars)
    assert len(content) <= max_chars
    json.loads(content)


def test_tiny_max_chars_returns_minimal_valid():
    """A tiny budget returns a minimal valid JSON fragment."""
    content, truncated = truncate_for_model({"a": 1}, 2)
    assert truncated
    assert len(content) <= 2
    assert content in ('""', "[]", "{}", "0")


def test_zero_max_chars():
    """A zero budget returns an empty string."""
    content, truncated = truncate_for_model({"a": 1}, 0)
    assert truncated
    assert content == ""


def test_one_char_max():
    """With a one-character budget, the best valid JSON is '0' (not an empty string)."""
    content, truncated = truncate_for_model("hello", 1)
    assert truncated
    assert content == "0"
    assert json.loads(content) == 0
