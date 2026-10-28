# Truncation Pipeline

This document describes how `truncate_for_model(data, max_chars)` reduces an arbitrary Python value into a **valid JSON string no longer than `max_chars` characters**, while preserving as much structure and information as possible.

The pipeline has two stages:

1. **Stage 1 — Normalization**: turn any Python value into something JSON-serializable.
2. **Stage 2 — Reduction**: shrink the normalized value until it fits the character budget.

---

## Overview

```
              ┌─────────────────────┐
   input ───► │  Stage 1: _prepare  │ ───► normalized data ───► _dumps ──► fits?
              └─────────────────────┘                                         │
                                                                        yes ─┴─► return (json, False)
                                                                         no ─────► Stage 2: _reduce
                                                                                        │
                                                                              return (json, True)
```

- If the initial serialization already fits, **nothing is truncated** and the flag is `False`.
- Otherwise the **reduction stage** produces a smaller JSON and the flag is `True`.
- For `max_chars <= 0`, the result is `("", True)`.

---

## Stage 1 — Normalization (`_prepare`)

The goal is to convert arbitrary Python values into JSON-friendly primitives **before** any budgeting logic runs.

### Rules

| Input type | Result |
|---|---|
| `None`, `bool`, `int`, `str` | unchanged |
| finite `float` | unchanged |
| `NaN` / `±Inf` | replaced with `"nan"`, `"inf"`, `"-inf"` (strings) |
| other `numbers.Number` (`Decimal`, numpy scalars, `Fraction`, …) | converted to `int` if integral and safe, else `float`; `str` on failure |
| `bytes`, `bytearray`, `memoryview` | UTF-8 decoded with `errors="replace"` |
| `dict` | keys converted to `str`, values normalized recursively |
| `list`, `tuple` | normalized recursively into a `list` |
| `set`, `frozenset` | normalized recursively, then **sorted** deterministically |
| Pydantic v2 / v1 models | via `model_dump(mode="json")` / `model_dump()` / `dict()` |
| dataclasses | field-by-field `{name: value}` |
| anything else | `str(value)` |

### Cycle & depth guards

- **Cycles**: container ids are tracked in a `seen` set. A re-entered object becomes `"<cycle>"`.
- **Depth**: recursion deeper than `MAX_DEPTH` (50) becomes `"<max-depth>"`.

Both guards keep the output safe for arbitrarily deep or self-referential inputs.

---

## Stage 2 — Reduction (`_reduce`)

Reduction is **type-aware**: lists stay lists, dicts stay dicts, strings stay strings. The dispatch is:

| Normalized value | Reducer |
|---|---|
| `list` | `_reduce_list` |
| `dict` | `_reduce_dict` |
| `str` | `_reduce_str` |
| scalar (`bool`, `int`, `float`, `null`) | `_valid_fallback` if it does not fit |

### `_reduce_list` — drop trailing items

The list is truncated by keeping a **prefix** and appending a marker:

```json
{"_truncated": true, "returned": K, "total": N}
```

- A **binary search** finds the largest `K` whose serialization still fits `max_chars`.
- If the input list was **already truncated** (last item is a marker), the marker's `total` is preserved instead of being reset to the current length — repeated truncations accumulate correctly.
- If not even `[{"_truncated": true, "total": N}]` fits, the reducer falls back to `_valid_fallback`.

### `_reduce_str` — truncate a string

The string is shortened and a marker is appended:

```
"...[truncated <n> chars]"
```

The longest prefix that fits together with the marker is found by binary search.

### `_reduce_dict` — the most involved reducer

If the dict is too big, a **loop** tries actions in a fixed priority order on each iteration, always targeting the **fattest** (largest serialized) non-marker field:

#### A. Recurse into a nested container

If the fattest field is a `list` or `dict`:

1. Compute the **overhead** of everything else in the dict.
2. Give the nested value a budget of `max_chars - overhead`.
3. Recursively `_reduce` it and re-parse via `json.loads`.
4. Accept the result only if it is the **same type** and **strictly smaller**.

This keeps nested structure intact before resorting to dropping it.

#### B. Truncate a long string value

If the fattest field is a `str` longer than `MIN_STR_KEEP` (20):

- Remove at least `overflow` characters, plus `MARKER_RESERVE` (40) room for the marker.
- Append `"...[truncated <n> chars]"`.

#### C. Drop the field

If A and B did not help, remove the field entirely and record its name in the `_truncated_fields` list:

```json
{"_truncated_fields": ["some_long_field", "..."]}
```

- If `_truncated_fields` collides with an existing key (and the existing value is not already a list), the marker is renamed to `_truncated_fields_1`, `_2`, …

#### D. Compress the marker itself

If dropping data no longer helps and the `_truncated_fields` list is itself too big:

1. Halve it and append `"...[truncated <n> more fields]"`.
2. If still too big, keep only the first name.
3. If still too big, remove the marker entirely.

The loop is bounded by `max_iterations = max(100, len(dict) * 2 + 20)` so a pathological structure cannot spin forever. Each iteration either shrinks the serialization or breaks out.

### `_valid_fallback` — last resort

If no safe reduction fits, return the largest of `'"..."'`, `'""'`, `'[]'`, `'{}'`, `'0'`, `'null'` that fits. If none fit (i.e. `max_chars < 2`), return `""`.

**Guarantee**: the returned string is always valid JSON whenever `max_chars >= 2`.

---

## Decision flow (Stage 2)

```
_reduce(value, budget)
│
├─ list    ──► _reduce_list ──► find largest prefix + marker
│                                (preserve total if already truncated)
│
├─ dict    ──► _reduce_dict ──► loop:
│                                A. shrink fattest nested container
│                                B. truncate fattest long string
│                                C. drop fattest field → _truncated_fields
│                                D. compress / drop the marker itself
│                               fallback if no step makes progress
│
├─ str     ──► _reduce_str  ──► longest prefix + "...[truncated n chars]"
│
└─ scalar  ──► fits? yes → return as-is
                 no  → _valid_fallback
```

---

## What the caller sees

| Situation | Output | Flag |
|---|---|---|
| Original fits | original JSON | `False` |
| Truncated list | `[..., {"_truncated": true, "returned": K, "total": N}]` | `True` |
| Truncated dict | dict + `_truncated_fields` (or `_truncated_fields_N`) | `True` |
| Truncated string | `"prefix...[truncated n chars]"` | `True` |
| Budget too small for any marker | `'"..."'` / `'""'` / `'[]'` / `'{}'` / `'0'` / `'null'` / `""` | `True` |

**Downstream code should:**
- Treat the last item of a list as a possible marker, not user data.
- Treat `_truncated_fields` / `_truncated_fields_N` as metadata, not payload.

---

## Priorities at a glance

When shrinking a container, the pipeline prefers losing information in the **least destructive** order:

1. **Whole trailing list items** — cheap, structure preserved.
2. **Long string tails** — content preserved, just shortened.
3. **Nested container recursion** — recursion applies the same rules one level deeper.
4. **Whole dict fields** — names are recorded in `_truncated_fields`.
5. **The marker itself** — only after everything else failed.

Every step keeps the output **valid JSON** and preserves the **outer type** whenever the budget allows.