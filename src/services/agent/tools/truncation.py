import dataclasses
import json
import math
import numbers
from typing import Any

TRUNCATED_LIST_KEY = "_truncated"
TRUNCATED_FIELDS_KEY = "_truncated_fields"
CYCLE_MARKER = "<cycle>"
DEPTH_MARKER = "<max-depth>"

MIN_STR_KEEP = 20
MARKER_RESERVE = 40
MAX_DEPTH = 50


def truncate_for_model(data: Any, max_chars: int | None) -> tuple[str, bool]:
    """
    Serialize `data` into a string no longer than `max_chars`.

    Structure is preserved where possible: whole list items are dropped first, then long strings are shortened,
    then dictionary fields are dropped, and only after that are the markers themselves compressed.
    Field type is preserved where possible: a list stays a list, a dictionary stays a dictionary.
    Fallback to `_valid_fallback` is used only when the budget is smaller than the minimum marker.

    Args:
        data (Any): Arbitrary Python value to serialize.
        max_chars (int): Maximum length of the returned string.

    Returns:
        tuple[str, bool]: The serialized content and a flag indicating
            whether truncation occurred.

    Guarantees:
        - content is valid JSON when max_chars >= 2 (otherwise an empty string).
        - Structure is preserved where possible (see above).

    Marker formats:
        - A truncated list ends with an object
          `{"_truncated": true, "returned": K, "total": N}`.
          Downstream code must account for the last item possibly being a
          marker rather than user data.
        - A dictionary with dropped fields has a `_truncated_fields` key
          containing the names of dropped fields (on collision with input
          data, `_truncated_fields_1`, `_2`, ... are used).

    Limitations:
        - `Decimal`, `numpy` numbers, and other `numbers.Number` instances
          are converted to `int`/`float`; precision may be lost (for example,
          `Decimal("0.1") + Decimal("0.2")`).
        - `bytes` are decoded as UTF-8 with `errors="replace"`.
        - `set`/`frozenset` are converted to a sorted list (deterministically).
        - `NaN`/`±Inf` are replaced with the strings `"nan"`, `"inf"`, `"-inf"`.
    """
    if max_chars is not None and max_chars <= 0:
        return "", True

    prepared = _prepare(data)
    serialized = _dumps(prepared)
    if max_chars is None or len(serialized) <= max_chars:
        return serialized, False

    return _reduce(prepared, max_chars)


# Stage 1: normalization of arbitrary Python values
def _prepare(data: Any) -> Any:
    """
    Normalize arbitrary Python values into JSON-serializable primitives.

    Args:
        data (Any): Arbitrary Python value to normalize.

    Returns:
        Any: A JSON-serializable normalized representation.
    """
    return _prepare_impl(data, seen=set(), depth=0)


def _prepare_impl(data: Any, seen: set[int], depth: int) -> Any:
    """
    Recursively normalize `data`, tracking seen container ids and depth.

    Args:
        data (Any): Value to normalize.
        seen (set[int]): Ids of containers currently being processed, used to detect cycles.
        depth (int): Current recursion depth.

    Returns:
        Any: A JSON-serializable normalized representation.
    """
    if depth > MAX_DEPTH:
        return DEPTH_MARKER

    if data is None or isinstance(data, bool):
        return data
    if isinstance(data, int):
        return data
    if isinstance(data, float):
        return data if math.isfinite(data) else str(data)
    if isinstance(data, str):
        return data

    # Exotic numeric types: numpy scalars, Decimal, Fraction, ...
    if isinstance(data, numbers.Number) and not isinstance(data, complex):
        try:
            as_float = float(data)
            if not math.isfinite(as_float):
                return str(as_float)
            if as_float.is_integer() and abs(as_float) < 2 ** 63:
                return int(as_float)
            return as_float
        except (TypeError, ValueError, OverflowError):
            return str(data)

    # Leaves that do not require cycle tracking
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data).decode("utf-8", errors="replace")

    # Track only containers/custom objects here; scalar leaves were handled above
    obj_id = id(data)
    if obj_id in seen:
        return CYCLE_MARKER

    seen.add(obj_id)
    try:
        if isinstance(data, dict):
            return {str(k): _prepare_impl(v, seen, depth + 1) for k, v in data.items()}

        if isinstance(data, (list, tuple)):
            return [_prepare_impl(v, seen, depth + 1) for v in data]

        if isinstance(data, (set, frozenset)):
            prepared = [_prepare_impl(v, seen, depth + 1) for v in data]
            return sorted(prepared, key=_dumps)

        # Pydantic v2 / v1
        for method_name, kwargs in (("model_dump", {"mode": "json"}), ("model_dump", {}), ("dict", {})):
            method = getattr(data, method_name, None)
            if callable(method):
                try:
                    return _prepare_impl(method(**kwargs), seen, depth + 1)
                except (TypeError, ValueError, AttributeError):
                    continue

        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            try:
                return {
                    f.name: _prepare_impl(getattr(data, f.name), seen, depth + 1)
                    for f in dataclasses.fields(data)
                }
            except (AttributeError, TypeError):
                pass

        return str(data)
    finally:
        seen.discard(obj_id)


# Stage 2: reduction to the character budget

def _reduce(data: Any, max_chars: int, depth: int = 0) -> tuple[str, bool]:
    """
    Reduce normalized data so its JSON representation fits `max_chars`.

    Args:
        data (Any): Normalized data (output of `_prepare`).
        max_chars (int): Maximum length of the resulting JSON string.
        depth (int): Current recursion depth.

    Returns:
        tuple[str, bool]: The reduced JSON string and a truncation flag.
    """
    if depth > MAX_DEPTH:
        return _valid_fallback(max_chars)
    if isinstance(data, list):
        return _reduce_list(data, max_chars)
    if isinstance(data, dict):
        return _reduce_dict(data, max_chars, depth)
    if isinstance(data, str):
        return _reduce_str(data, max_chars)

    # Scalar: truncation would make the JSON invalid (True/False/None/float)
    text = _dumps(data)
    if len(text) <= max_chars:
        return text, False
    return _valid_fallback(max_chars)


def _reduce_list(items: list, max_chars: int) -> tuple[str, bool]:
    """
    Binary search for how many leading items to keep.

    If the input list is already truncated (the last item is a `{_truncated: true, ...}` marker), the original `total`
    is preserved instead of being overwritten with the current size.

    Args:
        items (list): The list to reduce.
        max_chars (int): Maximum length of the resulting JSON string.

    Returns:
        tuple[str, bool]: The reduced JSON string and a truncation flag.
    """
    real_items = items
    original_total = len(items)

    if items and isinstance(items[-1], dict) and items[-1].get(TRUNCATED_LIST_KEY) is True:
        original_total = items[-1].get("total", len(items) - 1)
        real_items = items[:-1]

    def make(k: int) -> str:
        marker = {TRUNCATED_LIST_KEY: True, "returned": k, "total": original_total}
        return _dumps(real_items[:k] + [marker])

    # Find the largest prefix length that still fits with the truncation marker
    lo, hi = 0, len(real_items)
    best: tuple[str, bool] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = make(mid)
        if len(candidate) <= max_chars:
            best = (candidate, True)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    # Preserve list type even in fallback when possible
    minimal = _dumps([{TRUNCATED_LIST_KEY: True, "total": original_total}])
    if len(minimal) <= max_chars:
        return minimal, True
    return _valid_fallback(max_chars)


def _reduce_dict(d: dict, max_chars: int, depth: int = 0) -> tuple[str, bool]:
    """
    Iteratively reduce a dictionary.

    The reduction tries, in order:
        A. Recursively shrink a nested container (preserving its type).
        B. Truncate a long string value.
        C. Drop a field (record it in `_truncated_fields`).
        D. Compress the `_truncated_fields` marker if it becomes the problem.

    Args:
        d (dict): The dictionary to reduce.
        max_chars (int): Maximum length of the resulting JSON string.
        depth (int): Current recursion depth.

    Returns:
        tuple[str, bool]: The reduced JSON string and a truncation flag.
    """
    working = dict(d)
    truncated = False

    marker_key = TRUNCATED_FIELDS_KEY
    existing = working.get(marker_key)
    if existing is not None and not isinstance(existing, list):
        i = 1
        while f"{TRUNCATED_FIELDS_KEY}_{i}" in working:
            i += 1
        marker_key = f"{TRUNCATED_FIELDS_KEY}_{i}"

    # Bound the loop so a pathological structure cannot spin forever
    max_iterations = max(100, len(working) * 2 + 20)

    for _ in range(max_iterations):
        serialized = _dumps(working)
        if len(serialized) <= max_chars:
            return serialized, truncated

        non_marker_keys = [k for k in working if k != marker_key]
        made_progress = False

        # --- A / B / C: only when there are non-marker fields -------------
        if non_marker_keys:
            fattest_key = max(non_marker_keys, key=lambda k: len(_dumps(working[k])))
            fattest_val = working[fattest_key]
            field_size = len(_dumps(fattest_val))

            # A: recurse into a nested container
            if isinstance(fattest_val, (list, dict)):
                overhead = len(serialized) - field_size
                # Reserve the rest of the serialized object; the nested value gets the remaining budget
                budget = max_chars - overhead
                if budget > MARKER_RESERVE:
                    reduced_str, _ = _reduce(fattest_val, budget, depth + 1)
                    try:
                        new_val = json.loads(reduced_str)
                    except json.JSONDecodeError:
                        new_val = None
                    if (
                            new_val is not None
                            and isinstance(new_val, type(fattest_val))
                            and len(_dumps(new_val)) < field_size
                    ):
                        working[fattest_key] = new_val
                        truncated = True
                        made_progress = True

            # B: truncate a long string
            if not made_progress and isinstance(fattest_val, str) and len(fattest_val) > MIN_STR_KEEP:
                overflow = len(serialized) - max_chars
                # Remove at least enough characters to cover the overflow, then leave room for the marker
                target = max(MIN_STR_KEEP, len(fattest_val) - overflow - MARKER_RESERVE)
                if target < len(fattest_val):
                    new_val = (fattest_val[:target] + f"...[truncated {len(fattest_val) - target} chars]")
                    if len(new_val) < len(fattest_val):
                        working[fattest_key] = new_val
                        truncated = True
                        made_progress = True

            # C: drop a field
            if not made_progress and len(working) > 1:
                working.pop(fattest_key)
                prev = working.get(marker_key)
                if isinstance(prev, list):
                    working[marker_key] = prev + [fattest_key]
                else:
                    working[marker_key] = [fattest_key]
                truncated = True
                made_progress = True

        # --- D: compress an oversized marker ---------------------------
        if not made_progress:
            marker_val = working.get(marker_key)
            if isinstance(marker_val, list):
                if len(marker_val) > 1:
                    keep = max(1, len(marker_val) // 2)
                    suffix = f"...[truncated {len(marker_val) - keep} more fields]"
                    new_marker = marker_val[:keep] + [suffix]
                    if len(_dumps(new_marker)) < len(_dumps(marker_val)):
                        working[marker_key] = new_marker
                        truncated = True
                        made_progress = True

                if not made_progress and len(marker_val) > 1:
                    # Last resort: keep only the first name
                    working[marker_key] = marker_val[:1]
                    truncated = True
                    made_progress = True

                if not made_progress:
                    # Does not fit at all — remove the marker entirely
                    del working[marker_key]
                    truncated = True
                    made_progress = True

        if not made_progress:
            break

    serialized = _dumps(working)
    if len(serialized) <= max_chars:
        return serialized, truncated

    return _valid_fallback(max_chars)


def _reduce_str(s: str, max_chars: int) -> tuple[str, bool]:
    """
    Reduce a string value to fit `max_chars` by truncating it.

    Args:
        s (str): The string to reduce.
        max_chars (int): Maximum length of the resulting JSON string.

    Returns:
        tuple[str, bool]: The reduced JSON string and a truncation flag.
    """
    serialized = _dumps(s)
    if len(serialized) <= max_chars:
        return serialized, False

    # Find the longest prefix that fits together with the truncation marker
    lo, hi = 0, len(s)
    best: tuple[str, bool] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        marker = f"...[truncated {len(s) - mid} chars]"
        candidate = _dumps(s[:mid] + marker)
        if len(candidate) <= max_chars:
            best = (candidate, True)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    return _valid_fallback(max_chars)


def _valid_fallback(max_chars: int) -> tuple[str, bool]:
    """
    Return the largest guaranteed-valid JSON that fits within `max_chars`.

    Args:
        max_chars (int): Maximum length of the returned string.

    Returns:
        tuple[str, bool]: A valid JSON fragment and `True` (always truncated).
    """
    for fb in ('"..."', '""', "[]", "{}", "0", "null"):
        if len(fb) <= max_chars:
            return fb, True

    return "", True


def _dumps(obj: Any) -> str:
    """
    Serialize to JSON with non-ASCII preserved and a string fallback.

    Args:
        obj (Any): The object to serialize.

    Returns:
        str: The JSON representation of `obj`.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj), ensure_ascii=False)
