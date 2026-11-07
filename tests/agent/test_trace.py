"""Lifecycle tests for Trace: close, prune, size cap, unclosed warning, context manager.

CONTRACT: services/agent/trace.py is PII/retention critical. Its documented
lifecycle behaviors (idempotent close, best-effort prune by count and TTL,
bounded file size with a single truncated_at_bytes marker, ResourceWarning
for a GC'd unclosed trace, unexpected_error on context-manager exit) are
asserted here. Record-content tests (step/finalize/tool field shapes) live
in agent/test_tool_registry.py and agent/nodes/test_finalize.py.
"""
import gc
import os
import re
import time
import warnings
from pathlib import Path

import pytest

import services.agent.trace as trace_mod
from services.agent.trace import Trace


# ---------------------------------------------------------------------------
# Lifecycle: close / context manager
# ---------------------------------------------------------------------------

def test_close_is_idempotent(tmp_path):
    """close() is idempotent: a second call neither raises nor flips ``closed`` back."""
    t = Trace(tmp_path)
    t.close()
    assert t.closed is True

    t.close()  # must not raise
    assert t.closed is True


def test_write_after_close_is_noop(tmp_path):
    """_write() after close() is a no-op — the file size does not change."""
    t = Trace(tmp_path)
    t.close()

    size_before = t.path.stat().st_size
    t._write({'t': 'step', 'n': 1})
    size_after = t.path.stat().st_size

    assert size_after == size_before


def test_close_swallows_handler_close_error(tmp_path):
    """A handler close failure must not propagate — the agent error, if any, must not be masked.

    Also asserts that _finalizer.detach() runs before _fh.close(): if the
    detach were moved after the try/except, the finalizer would stay alive
    and emit a ResourceWarning at GC. The ``alive is False`` assertion pins
    that ordering.
    """
    t = Trace(tmp_path)
    real_handle = t._fh  # real io.TextIOWrapper — must be closed in finally

    class _FailingHandle:
        def close(self) -> None:
            raise OSError('simulated close failure')

    t._fh = _FailingHandle()  # type: ignore[assignment]
    try:
        t.close()  # must not raise
        assert t.closed is True
        assert t._finalizer.alive is False
    finally:
        real_handle.close()  # release the leaked FD; the stub holds none


def test_context_manager_closes_on_normal_exit(tmp_path):
    """__exit__ closes the trace on a clean exit; no records are written, so records() stays empty."""
    with Trace(tmp_path) as t:
        assert t.closed is False

    assert t.closed is True
    assert t.records() == []


def test_context_manager_close_after_explicit_close(tmp_path):
    """Calling close() inside the with block makes the implicit close in __exit__ a no-op."""
    with Trace(tmp_path) as t:
        t.close()
        assert t.closed is True

    # __exit__ called close() again — idempotent, no error.
    assert t.closed is True


def test_close_after_context_manager(tmp_path):
    """close() after the with block is idempotent — the trace stays closed."""
    t = Trace(tmp_path)
    with t:
        pass

    assert t.closed is True
    t.close()  # idempotent
    assert t.closed is True


# ---------------------------------------------------------------------------
# run_id
# ---------------------------------------------------------------------------

def test_run_id_matches_documented_shape(tmp_path):
    """run_id is 'YYYYMMDD_HHMMSS_<hex8>' — safe as a filename."""
    t = Trace(tmp_path)
    try:
        assert re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{8}', t.run_id)
    finally:
        t.close()


def test_run_id_matches_filename_stem(tmp_path):
    """trace.run_id equals the trace file stem (public property contract)."""
    t = Trace(tmp_path)
    try:
        assert t.path.stem == t.run_id
    finally:
        t.close()


def test_all_records_share_one_run_id(tmp_path):
    """Every record in one file carries the same run_id — the trace is one coherent run."""
    t = Trace(tmp_path)
    try:
        t._write({'t': 'a'})
        t._write({'t': 'b'})
        t._write({'t': 'c'})
        assert all(r['run_id'] == t.run_id for r in t.records())
    finally:
        t.close()


# ---------------------------------------------------------------------------
# Prune by count (MAX_TRACE_FILES)
# ---------------------------------------------------------------------------

def test_prune_by_count_keeps_most_recent(tmp_path, monkeypatch):
    """At most MAX_TRACE_FILES *.jsonl survive; the oldest by st_mtime are removed.

    The new trace participates in 'most recent' ordering: it is created
    before pruning, so the oldest five of (5 pre-existing + 1 new) are
    pruned by the MAX_TRACE_FILES=3 threshold, leaving exactly 3 files.
    """
    monkeypatch.setattr(trace_mod, 'MAX_TRACE_FILES', 3)
    monkeypatch.setattr(trace_mod, 'TRACE_TTL_DAYS', 3650)  # effectively off

    base = time.time() - 10_000
    for i in range(5):
        p = tmp_path / f'20240101_0000{i:02d}_abcd0000.jsonl'
        p.write_text('{}\n')
        os.utime(p, (base + i * 100, base + i * 100))

    t = Trace(tmp_path)  # creates a new file, then prunes
    try:
        remaining = {p.name for p in tmp_path.glob('*.jsonl')}

        assert len(remaining) == 3  # MAX_TRACE_FILES
        assert '20240101_000000_abcd0000.jsonl' not in remaining  # oldest
        assert '20240101_000001_abcd0000.jsonl' not in remaining  # 2nd oldest
        assert '20240101_000002_abcd0000.jsonl' not in remaining  # 3rd oldest
        assert '20240101_000003_abcd0000.jsonl' in remaining
        assert '20240101_000004_abcd0000.jsonl' in remaining
        assert t.path.name in remaining
    finally:
        t.close()


def test_prune_ignores_non_jsonl_files(tmp_path, monkeypatch):
    """Prune only touches *.jsonl — sibling *.txt / *.log files survive."""
    monkeypatch.setattr(trace_mod, 'MAX_TRACE_FILES', 1)
    monkeypatch.setattr(trace_mod, 'TRACE_TTL_DAYS', 3650)

    non_jsonl = tmp_path / 'notes.txt'
    non_jsonl.write_text('keep me\n')

    stale_jsonl = tmp_path / '20240101_000000_abcd0000.jsonl'
    stale_jsonl.write_text('{}\n')
    os.utime(stale_jsonl, (0, 0))  # ancient — pruned regardless of count/TTL

    t = Trace(tmp_path)
    try:
        remaining_jsonl = {p.name for p in tmp_path.glob('*.jsonl')}
        assert remaining_jsonl == {t.path.name}
        assert non_jsonl.exists()  # not touched by prune
    finally:
        t.close()


# ---------------------------------------------------------------------------
# Prune by TTL (TRACE_TTL_DAYS)
# ---------------------------------------------------------------------------

def test_prune_by_ttl_removes_old_files(tmp_path, monkeypatch):
    """A file older than TRACE_TTL_DAYS is removed even well under MAX_TRACE_FILES."""
    monkeypatch.setattr(trace_mod, 'TRACE_TTL_DAYS', 7)
    monkeypatch.setattr(trace_mod, 'MAX_TRACE_FILES', 500)

    old = tmp_path / '20240101_000000_abcd0000.jsonl'
    old.write_text('{}\n')
    old_mtime = time.time() - 8 * 86400
    os.utime(old, (old_mtime, old_mtime))

    t = Trace(tmp_path)
    try:
        assert not old.exists()
    finally:
        t.close()


def test_prune_uses_mtime_not_run_id(tmp_path, monkeypatch):
    """TTL is based on st_mtime, not the timestamp embedded in run_id.

    A file restored from backup may carry an ancient name but a fresh
    mtime — it must survive, per the documented carve-out in _prune_old_runs.
    """
    monkeypatch.setattr(trace_mod, 'TRACE_TTL_DAYS', 7)
    monkeypatch.setattr(trace_mod, 'MAX_TRACE_FILES', 500)

    old_named = tmp_path / '19700101_000000_deadbeef.jsonl'
    old_named.write_text('{}\n')  # mtime ≈ now, despite the epoch-style name

    t = Trace(tmp_path)
    try:
        assert old_named.exists()
    finally:
        t.close()


def test_prune_swallows_unlink_error(tmp_path, monkeypatch):
    """Prune is best-effort: an unlink() failure must not abort trace creation.

    The unlink patch is selective (only ``stale`` raises) so the test does
    not break if a future change introduces unlink() calls elsewhere in
    Trace (e.g. in close()); those calls would reach the real
    Path.unlink and succeed.
    """
    monkeypatch.setattr(trace_mod, 'TRACE_TTL_DAYS', 7)
    monkeypatch.setattr(trace_mod, 'MAX_TRACE_FILES', 500)

    stale = tmp_path / '20240101_000000_abcd0000.jsonl'
    stale.write_text('{}\n')
    os.utime(stale, (0, 0))  # ancient — a prune target

    real_unlink = Path.unlink

    def selective_unlink(self, missing_ok=False):
        if self == stale:
            raise OSError('simulated unlink failure')
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, 'unlink', selective_unlink)

    # Must not raise.
    t = Trace(tmp_path)
    try:
        assert stale.exists()  # unlink failed, file survived
    finally:
        t.close()


# ---------------------------------------------------------------------------
# Size cap (MAX_FILE_BYTES)
# ---------------------------------------------------------------------------

def test_size_cap_emits_single_marker(tmp_path, monkeypatch):
    """Overflow writes exactly one truncated_at_bytes marker; further writes are dropped.

    Also pins that the records before the cap actually landed: two step
    records fit under the 500-byte budget (~191 bytes each), the third
    overflows. Without the step-records assertion, a _write that wrote
    nothing (but still emitted the marker) would pass this test.
    """
    monkeypatch.setattr(trace_mod, 'MAX_FILE_BYTES', 500)
    t = Trace(tmp_path)
    try:
        for i in range(50):
            t._write({'t': 'step', 'n': i, 'data': 'x' * 100})

        records = t.records()
        markers = [r for r in records if r['t'] == 'truncated_at_bytes']
        assert len(markers) == 1
        assert markers[0]['limit_bytes'] == 500

        step_records = [r for r in records if r['t'] == 'step']
        assert len(step_records) == 2  # exactly two fit before the cap
    finally:
        t.close()


def test_writes_after_cap_are_noop(tmp_path, monkeypatch):
    """After the size cap is hit, no further bytes reach the file — not even a second marker."""
    monkeypatch.setattr(trace_mod, 'MAX_FILE_BYTES', 500)
    t = Trace(tmp_path)
    try:
        for i in range(20):
            t._write({'t': 'step', 'n': i, 'data': 'x' * 100})

        size_at_cap = t.path.stat().st_size
        records_at_cap = t.records()
        assert records_at_cap[-1]['t'] == 'truncated_at_bytes'

        for i in range(10):
            t._write({'t': 'step', 'n': i, 'data': 'y' * 100})

        assert t.path.stat().st_size == size_at_cap
        assert t.records() == records_at_cap
    finally:
        t.close()


def test_records_reads_truncated_file(tmp_path, monkeypatch):
    """records() parses a truncated file — the last record is the marker.

    Also verifies the documented size carve-out: the trailing
    truncated_at_bytes marker may push the file a few bytes past
    MAX_FILE_BYTES, so the assertion is ``>=``, not ``==``.
    """
    monkeypatch.setattr(trace_mod, 'MAX_FILE_BYTES', 300)
    t = Trace(tmp_path)
    try:
        for i in range(20):
            t._write({'t': 'step', 'n': i})

        records = t.records()
        assert isinstance(records, list)
        assert records[-1]['t'] == 'truncated_at_bytes'
        assert records[-1]['limit_bytes'] == 300
        assert t.path.stat().st_size >= 300  # cap reached
    finally:
        t.close()


def test_write_exactly_at_max_bytes_is_allowed(tmp_path, monkeypatch):
    """A write that lands exactly on MAX_FILE_BYTES passes — the cap check is strict ``>``.

    Guards against an off-by-one regression (> vs >=): if the cap fires on
    exact equality, the second identical write is dropped and the test
    fails with a truncated_at_bytes marker in the record list.
    """
    t = Trace(tmp_path)
    try:
        t._write({'t': 'probe'})
        one_record_size = t.path.stat().st_size

        monkeypatch.setattr(trace_mod, 'MAX_FILE_BYTES', one_record_size * 2)

        t._write({'t': 'probe'})  # exactly at boundary — must pass
        records = t.records()
        assert [r['t'] for r in records] == ['probe', 'probe']
        assert not any(r['t'] == 'truncated_at_bytes' for r in records)
    finally:
        t.close()


# ---------------------------------------------------------------------------
# Unclosed warning (weakref.finalize)
# ---------------------------------------------------------------------------

def test_unclosed_trace_emits_resource_warning(tmp_path):
    """A GC'd unclosed Trace emits a ResourceWarning — diagnostic only, no cleanup guarantee.

    The Trace is built inside a nested function so that the local variable
    cannot be kept alive by the test frame's f_locals (pytest can retain
    references across ``del``). The assertion is bound to the specific
    trace path: an unrelated unclosed Trace collected by ``gc.collect()``
    must not turn this test green.
    """
    path_holder: dict[str, Path] = {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')

        def _make_unclosed() -> None:
            t = Trace(tmp_path)
            path_holder['path'] = t.path
            # t leaves scope here → refcount drops to 0 → finalizer fires

        _make_unclosed()
        gc.collect()
        gc.collect()

    assert any(
        issubclass(w.category, ResourceWarning)
        and 'was not closed' in str(w.message)
        and str(path_holder['path']) in str(w.message)
        for w in caught
    ), f"expected ResourceWarning for {path_holder['path']}, got {[str(w.message) for w in caught]}"


# ---------------------------------------------------------------------------
# __exit__ on exception
# ---------------------------------------------------------------------------

def test_exit_with_exception_writes_unexpected_error(tmp_path):
    """An unhandled exception writes {'t': 'unexpected_error', 'kind', 'message'}."""
    with pytest.raises(ValueError):
        with Trace(tmp_path) as t:
            raise ValueError('boom')

    records = t.records()
    unexpected = [r for r in records if r['t'] == 'unexpected_error']
    assert len(unexpected) == 1
    assert unexpected[0]['kind'] == 'ValueError'
    assert 'boom' in unexpected[0]['message']


def test_exit_truncates_long_exception_message(tmp_path):
    """The diagnostic message is capped at 500 characters."""
    long_msg = 'x' * 1000
    with pytest.raises(RuntimeError, match='x{10}'):
        with Trace(tmp_path) as t:
            raise RuntimeError(long_msg)

    unexpected = [r for r in t.records() if r['t'] == 'unexpected_error']
    assert len(unexpected) == 1
    assert unexpected[0]['kind'] == 'RuntimeError'
    assert unexpected[0]['message'] == 'x' * 500
    assert len(unexpected[0]['message']) == 500


def test_exit_does_not_mask_original_exception(tmp_path):
    """__exit__ returns None: the original exception type propagates unchanged."""

    class CustomError(Exception):
        pass

    with pytest.raises(CustomError):
        with Trace(tmp_path) as t:
            raise CustomError('original')

    assert t.closed is True


def test_exit_skips_write_when_already_closed(tmp_path, monkeypatch):
    """__exit__ must short-circuit before calling _write when the trace is already closed.

    A behavioural assertion (``_write`` never called) is used instead of
    inspecting records: if the ``and not self._closed`` guard is removed,
    ``__exit__`` reaches ``_write`` and the spy catches the call — even
    though ``_write`` itself would then no-op on ``self._closed``.
    """
    t = Trace(tmp_path)
    write_calls = []
    real_write = t._write

    def spy(rec):
        write_calls.append(rec)
        return real_write(rec)

    monkeypatch.setattr(t, '_write', spy)

    t.close()
    with pytest.raises(RuntimeError):
        with t:
            raise RuntimeError('boom')

    assert write_calls == []  # __exit__ never reached _write


# ---------------------------------------------------------------------------
# records()
# ---------------------------------------------------------------------------

def test_records_on_empty_file_returns_empty_list(tmp_path):
    """A freshly created trace has no records — records() returns []."""
    t = Trace(tmp_path)
    try:
        assert t.records() == []
    finally:
        t.close()


def test_ts_is_iso8601_utc(tmp_path):
    """The ``ts`` field is ISO-8601 UTC with a 'Z' suffix — parseable across tools."""
    t = Trace(tmp_path)
    try:
        t._write({'t': 'x'})
        ts = t.records()[0]['ts']
        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', ts)
    finally:
        t.close()
