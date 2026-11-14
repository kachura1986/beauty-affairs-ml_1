import hashlib
import json
import secrets
import time
import weakref
from pathlib import Path
from typing import Any

from services.llm import Reply, ToolCall

from .domain import Budget, Finalization, TerminationReason
from .tools import ToolResult

# How many trace files to keep in RUNS_DIR.
MAX_TRACE_FILES = 500

# Hard cap on a single trace file: once exceeded, further records are dropped
# and a single `truncated_at_bytes` marker is written. Bounds per-run disk usage.
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# Trace files older than this are pruned on every new trace creation, in
# addition to the MAX_TRACE_FILES count cap. Bounds on-disk retention over time.
TRACE_TTL_DAYS = 7

# Fields injected by _write() itself; a payload must not contain them —
# **rec would otherwise silently overwrite. Extend explicitly if a future
# record needs one of these names.
_RESERVED_FIELDS = frozenset({'run_id', 'agent', 'ts'})


class Trace:
    """
    Records an agent_graph run (input, every LLM call, final result) as JSONL, one record per line.

    CONTRACT (PII and retention):
        - Tool arguments are recorded as ``args_hash`` + ``args_keys``, never raw values.
        - User / assistant message text is never written: ``step()`` records ``window_hash`` /
          ``window_msgs``; ``finish()`` records only ``answer_chars``.
        - ``error()`` ``message`` must be a machine-readable code (``type(exc).__name__``, reason
          string), never ``str(exc)`` from a tool/user-facing exception.
        - Single carve-out: ``__exit__`` writes a truncated (≤500 chars) ``str(exc)`` for *unhandled*
          exceptions that escaped the agent loop — diagnostic only. Do not promote this pattern to a
          general error path; use ``error()`` with a machine-readable code.
        - Bounds: one file ≤ ``MAX_FILE_BYTES``; the directory is pruned by ``MAX_TRACE_FILES`` count
          and ``TRACE_TTL_DAYS`` age on every new trace creation.
        - Every regular record carries an ``agent`` field (the construction-time agent name, or
          ``null`` for ad-hoc traces). The ``truncated_at_bytes`` marker is the only exception —
          it carries ``run_id`` and ``ts`` only.

    Lifecycle: ``run_id`` is generated internally; close via :meth:`close` (idempotent) or as a
    context manager. An unclosed trace emits a ``ResourceWarning`` via :mod:`weakref.finalize`
    (diagnostic only, no cleanup guarantee). ``agent_name`` is fixed at construction.
    """

    def __init__(self, runs_dir: Path, *, agent_name: str | None = None) -> None:
        """
        Create a new trace.

        Args:
            runs_dir (Path): Directory where the trace files are stored.
            agent_name (str | None): Name of the agent that produces this trace; written to the ``agent`` field of
                every record. ``None`` for ad-hoc traces created outside :class:`AgentService`.
        """
        self._runs_dir = runs_dir
        self._runs_dir.mkdir(parents=True, exist_ok=True)

        self._agent_name = agent_name
        self._run_id = self.generate_run_id()
        self.path = self._runs_dir / f'{self._run_id}.jsonl'

        self._closed = False
        self._fh = self.path.open('a', encoding='utf-8')
        self._bytes_written = self.path.stat().st_size
        self._capped = False

        self._finalizer = weakref.finalize(self, self._warn_unclosed, self.path)

        # Prune after opening our file so it participates in "most recent" ordering.
        self._prune_old_runs()

    @staticmethod
    def generate_run_id() -> str:
        """
        Generate a new run identifier with a UTC timestamp prefix.

        Returns:
            str: Format ``YYYYMMDD_HHMMSS_<hex>`` — date, time, 8 hex chars. Safe as a filename.
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        suffix = secrets.token_hex(4)

        return f'{timestamp}_{suffix}'

    @staticmethod
    def _warn_unclosed(path: Path) -> None:
        """
        Emit a ``ResourceWarning`` for a trace that was garbage-collected unclosed.

        Args:
            path (Path): Trace file path, used in the warning message.
        """
        import warnings
        warnings.warn(f"Trace {path} was not closed", ResourceWarning, stacklevel=2)

    def _prune_old_runs(self) -> None:
        """
        Keep only the most recent ``MAX_TRACE_FILES`` trace files; drop files older than ``TRACE_TTL_DAYS``.

        CONTRACT: best-effort — pruning failure must not affect the current run. TTL uses ``st_mtime``, not ``run_id``
        (a restored file may survive past ``TRACE_TTL_DAYS``). Not synchronized across processes; ``unlink()`` on
        an open file is safe on POSIX and raises ``PermissionError`` on Windows (swallowed).
        """
        try:
            files = sorted(self._runs_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return

        cutoff = time.time() - TRACE_TTL_DAYS * 86400

        for i, stale in enumerate(files):
            try:
                if i >= MAX_TRACE_FILES or stale.stat().st_mtime < cutoff:
                    stale.unlink()
            except OSError:
                pass

    def close(self) -> None:
        """
        Close the underlying file handle.

        CONTRACT: idempotent and never raises — a close failure must not mask the original agent error.
        """
        if self._closed:
            return
        self._closed = True

        # Cancel the diagnostic finalizer: the trace was closed explicitly
        self._finalizer.detach()

        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has been called."""
        return self._closed

    @property
    def run_id(self) -> str:
        """Public run identifier; matches the trace file stem."""
        return self._run_id

    @property
    def agent_name(self) -> str | None:
        """Name of the agent that produced this trace (``None`` for ad-hoc runs).

        CONTRACT: read-only property; independent of ``_closed`` — safe to read after :meth:`close`.
        """
        return self._agent_name

    def records(self) -> list[dict[str, Any]]:
        """
        Read back all JSONL records. Helper for post-mortem and tests.

        CONTRACT: not for hot-path production use — reads the entire file (up to roughly
        ``MAX_FILE_BYTES`` plus the trailing ``truncated_at_bytes`` marker).

        Returns:
            list[dict[str, Any]]: Parsed records in file order.
        """
        return [
            json.loads(line)
            for line in self.path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def __enter__(self) -> 'Trace':
        """Enter the context manager; returns self."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Close the trace on context exit; record unhandled exceptions best-effort.

        CONTRACT: the diagnostic write never masks the original exception. Returning None propagates
        the exception unchanged.
        """
        if exc_type is not None and not self._closed:
            try:
                # Diagnostic only: unhandled exception escaped the agent loop, capped at 500 chars.
                # General error paths must use `error()` with a machine-readable code.
                self._write({
                    't': 'unexpected_error',
                    'kind': exc_type.__name__,
                    'message': str(exc)[:500],
                })
            except Exception:  # noqa: BLE001
                pass
        self.close()

    def _write(self, rec: dict[str, Any]) -> None:
        """
        Append one record as a JSONL line, enforcing the per-file size cap.

        CONTRACT: no-op when closed or capped. On overflow, writes a single
        ``truncated_at_bytes`` marker and drops all further records.

        RESERVED FIELDS: ``run_id``, ``agent``, ``ts`` are injected here and enforced — callers
        MUST NOT pass them in ``rec``. ``**rec`` would otherwise silently overwrite the injected
        values; extend ``_RESERVED_FIELDS`` explicitly if a payload needs one of these names.
        The ``truncated_at_bytes`` marker is written manually below and bypasses the check.

        Args:
            rec (dict[str, Any]): Record payload; ``run_id``, ``agent``, and ``ts`` are added automatically.

        Raises:
            ValueError: If ``rec`` contains any of ``_RESERVED_FIELDS``.
        """
        if self._closed:
            return

        if self._capped:
            return

        overlap = _RESERVED_FIELDS & rec.keys()
        if overlap:
            raise ValueError(
                f"Trace._write: payload contains reserved keys: {sorted(overlap)}; reserved: {sorted(_RESERVED_FIELDS)}"
            )

        rec = {
            'run_id': self._run_id,
            'agent': self._agent_name,
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            **rec,
        }

        line = json.dumps(rec, ensure_ascii=False, default=str) + '\n'
        line_bytes = len(line.encode())

        # Hard size cap: once exceeded, drop records and emit a single marker.
        if self._bytes_written + line_bytes > MAX_FILE_BYTES:
            self._capped = True
            marker = {
                'run_id': self._run_id,
                'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                't': 'truncated_at_bytes',
                'limit_bytes': MAX_FILE_BYTES,
            }
            self._fh.write(json.dumps(marker, ensure_ascii=False) + '\n')
            self._fh.flush()
            return

        self._bytes_written += line_bytes
        self._fh.write(line)
        self._fh.flush()

    @staticmethod
    def _hash(obj: Any) -> str:
        """
        Return a short stable hash of a JSON-serializable object.

        CONTRACT: deterministic, unsalted SHA-256 truncated to 12 hex chars — anti-PII measure
        for free-text user data, NOT a cryptographic commitment. Low-entropy values (``top_k``,
        ``min_price``) are recoverable by enumeration; high-entropy secrets are not reversible.

        Args:
            obj (Any): Object to hash; serialized with sorted keys and ``default=str``.

        Returns:
            str: First 12 hex chars of SHA-256.
        """
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)

        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def step(
            self,
            n: int,
            window: list[dict[str, Any]],
            reply: Reply,
            budget: Budget,
            latency_ms: int,
            *,
            system_prompt: str,
    ) -> None:
        """
        Record one LLM decision step.

        Args:
            n (int): Step number.
            window (list[dict[str, Any]]): Messages sent to the model; stored as ``window_hash`` + count.
            reply (Reply): Model reply (tokens, cost, tool calls).
            budget (Budget): Budget snapshot after the call.
            latency_ms (int): Call latency in milliseconds.
            system_prompt (str): Effective system prompt sent this step; stored as hash + length.
        """
        # Warning is emitted at most once per run, so `system_hash` changes at most once.
        self._write({
            't': 'step',
            'step': n,
            'model': reply.model,
            'in_tokens': reply.in_tokens,
            'out_tokens': reply.out_tokens,
            'cost': round(reply.cost, 6),
            'latency_ms': latency_ms,
            'window_hash': self._hash(window),
            'window_msgs': len(window),
            'system_hash': self._hash(system_prompt),
            'system_chars': len(system_prompt),
            'decision': 'tool_call' if reply.tool_calls else ('answer' if reply.text.strip() else 'empty'),
            'tools': [c['name'] for c in reply.tool_calls],
            'budget': budget.snapshot(),
        })

    def finalize(
            self,
            n: int,
            history: list[dict[str, Any]],
            reply: Reply,
            budget: Budget,
            latency_ms: int,
            *,
            system_prompt: str,
    ) -> None:
        # Mirrors `step`, but tags the record `t='finalize'` to separate the finalization call
        # from decision steps in post-mortem. No `window_hash` — finalization sends `history` as-is
        # (only its length is recorded; content is PII-sensitive).
        self._write({
            't': 'finalize',
            'step': n,
            'model': reply.model,
            'in_tokens': reply.in_tokens,
            'out_tokens': reply.out_tokens,
            'cost': round(reply.cost, 6),
            'latency_ms': latency_ms,
            'history_msgs': len(history),
            'system_hash': self._hash(system_prompt),
            'system_chars': len(system_prompt),
            'decision': 'tool_call' if reply.tool_calls else ('answer' if reply.text.strip() else 'empty'),
            'tools': [c['name'] for c in reply.tool_calls],
            'budget': budget.snapshot(),
        })

    def tool(self, n: int, call: ToolCall, res: ToolResult, ms: int) -> None:
        """
        Record one tool invocation.

        CONTRACT: args are stored as ``args_hash`` + ``args_keys`` only, never raw values.
        ``args_hash`` is deterministic SHA-256 truncated to 12 hex chars — low-entropy fields
        are recoverable by enumeration (required for post-mortem diffing; an HMAC would destroy
        that). ``hint`` is intentionally NOT written: for ``validation`` / ``extra_forbidden``
        it embeds the raw field name, which would re-introduce the leak ``args_keys`` prevents.

        Args:
            n (int): Step number the tool belongs to.
            call (ToolCall): Tool call (name + arguments).
            res (ToolResult): Tool result (status, error fields, truncation, redacted arg keys).
            ms (int): Duration in milliseconds.
        """
        args = call.get('arguments', {})
        self._write({
            't': 'tool',
            'step': n,
            'tool': call['name'],
            'args_hash': self._hash(args),
            'args_keys': list(res.args_keys),
            'status': res.status.value,

            'error_code': res.error.error_code.value if res.error else None,
            'message': res.error.message if res.error else None,
            'retryable': res.error.retryable if res.error else None,
            'retry_after': res.error.retry_after if res.error else None,

            'truncated': res.truncated,
            'bytes_out': len(res.content),
            'duration_ms': ms,
        })

    def error(self, step: int, kind: str, message: str, latency_ms: int, budget: Budget) -> None:
        """
        Record a non-fatal error event.

        CONTRACT: ``message`` must be a machine-readable code (type name, reason string, or
        ``ContextOverflow.code``), never ``str(exc)`` — that may contain PII. The only carve-out is
        ``__exit__`` (unhandled, diagnostic only). Log the detail separately.

        Args:
            step (int): Step number when the error occurred.
            kind (str): Error kind, e.g. ``'llm_error'``, ``'context_overflow'``.
            message (str): Machine-readable error code.
            latency_ms (int): Latency in milliseconds.
            budget (Budget): Current budget.
        """
        self._write({
            't': 'error',
            'step': step,
            'kind': kind,
            'message': message,
            'latency_ms': latency_ms,
            'budget': budget.snapshot(),
        })

    def finish(
            self,
            reason: TerminationReason,
            budget: Budget,
            answer: str | None,
            *,
            finalization: Finalization = Finalization.NONE,
    ) -> None:
        """
        Record the terminal run record.

        CONTRACT: only the answer's length (``answer_chars``) is stored — the answer text is not written to disk.

        Args:
            reason (TerminationReason): Why the run ended.
            budget (Budget): Final budget snapshot.
            answer (str | None): Final answer; only its length is stored.
            finalization (Finalization): How the answer was produced. Defaults to ``NONE``.
        """
        self._write({
            't': 'finish',
            'reason': reason.value,
            'budget': budget.snapshot(),
            'answer_chars': len(answer or ''),
            'finalization': finalization.value,
        })
