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

# Hard cap on a single trace file. Once exceeded, further records are dropped
# and a single `truncated_at_bytes` marker is written. Bounds disk usage per run.
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# Trace files older than this are removed on every new trace creation,
# in addition to the MAX_TRACE_FILES count cap. Bounds disk usage over time.
TRACE_TTL_DAYS = 7


class Trace:
    """
    Records an agent_graph run (input, every LLM call, final result) as JSONL, one record per line.

    Lifecycle:
        - ``run_id`` is generated internally (UTC timestamp + random suffix); file names are unique,
          time-ordered, and cannot escape RUNS_DIR.
        - Close via :meth:`close` or as a context manager; calling ``close()`` twice is safe.
        - If garbage-collected without being closed, a ``ResourceWarning`` is emitted via
          :mod:`weakref.finalize` (diagnostic only, no cleanup guarantee).
        - Old trace files are pruned on every new trace creation (at most ``MAX_TRACE_FILES`` most recent).

    PII and retention:
        - Tool arguments are recorded as ``args_hash`` + ``args_keys``, never raw values
          (arguments may contain user text; traces are retained on disk).
        - User / assistant message text is never written: ``step()`` records ``window_hash`` and
          ``window_msgs``; ``finish()`` records only ``answer_chars``.
        - ``error()`` ``message`` must be a machine-readable code (``type(exc).__name__``, reason
          string), not ``str(exc)`` from a tool/user-facing exception.
        - Single exception: ``__exit__`` writes a truncated (≤500 chars) ``str(exc)`` for *unhandled*
          exceptions that escaped the agent loop — diagnostic only, on an exceptional path. Do not
          promote this pattern to a general error path; use ``error()`` with a machine-readable code.
        - A single file is capped at ``MAX_FILE_BYTES``; the directory is pruned by ``MAX_TRACE_FILES``
          count and ``TRACE_TTL_DAYS`` age on every new trace creation.
    """

    def __init__(self, runs_dir: Path) -> None:
        """
        Create a new trace.

        Args:
            runs_dir (Path): Directory where the trace files are stored.
        """
        self._runs_dir = runs_dir
        self._runs_dir.mkdir(parents=True, exist_ok=True)

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
        Keep only the most recent ``MAX_TRACE_FILES`` trace files and drop files older than ``TRACE_TTL_DAYS``.
        Best-effort: any failure (including permission errors while deleting a stale file) must not affect the current run.

        Notes:
            - TTL is based on ``st_mtime``, not on the timestamp embedded in ``run_id``:
              a file restored from backup may have a fresh ``mtime`` and therefore survive
              past ``TRACE_TTL_DAYS``. Accepted as an edge case for the minimal policy.
            - Prune is not synchronized across processes. ``unlink()`` on an open file is
              safe on POSIX (the descriptor stays valid) and raises ``PermissionError`` on
              Windows, which is swallowed by ``except OSError`` below.
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

        Idempotent. Never raises: a failure to close the file handle must not mask the original error from the agent run.
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

    def records(self) -> list[dict[str, Any]]:
        """
        Read back all JSONL records. Helper for post-mortem and tests.

        Not for hot-path production use: reads the entire file into memory (up to roughly ``MAX_FILE_BYTES``;
        the trailing ``truncated_at_bytes`` marker may push the file a few bytes past the limit).

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

        The diagnostic write never masks the original exception. Returning None propagates the exception unchanged.
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

        No-op when the trace is closed or already capped. On overflow, writes a single
        ``truncated_at_bytes`` marker and drops all further records.

        Args:
            rec (dict[str, Any]): Record payload; ``run_id`` and ``ts`` are added automatically.
        """
        if self._closed:
            return

        if self._capped:
            return

        rec = {
            'run_id': self._run_id,
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
            'cost_usd': round(reply.cost_usd, 6),
            'latency_ms': latency_ms,
            'window_hash': self._hash(window),
            'window_msgs': len(window),
            'system_hash': self._hash(system_prompt),
            'system_chars': len(system_prompt),
            'decision': 'tool_call' if reply.tool_calls else ('answer' if reply.text.strip() else 'empty'),
            'tools': [c['name'] for c in reply.tool_calls],
            'budget': budget.snapshot(),
        })

    def tool(self, n: int, call: ToolCall, res: ToolResult, ms: int) -> None:
        """
        Record one tool invocation.

        Args are stored as ``args_hash`` + ``args_keys`` only — never raw values (PII;
        files are retained on disk). The hash correlates identical calls but cannot be reversed.

        Args:
            n (int): Step number the tool belongs to.
            call (ToolCall): Tool call (name + arguments).
            res (ToolResult): Tool result (status, error fields, truncation).
            ms (int): Duration in milliseconds.
        """
        args = call.get('arguments', {})
        self._write({
            't': 'tool',
            'step': n,
            'tool': call['name'],
            'args_hash': self._hash(args),
            'args_keys': sorted(args.keys()) if isinstance(args, dict) else [],
            'status': res.status.value,

            'error_code': res.error.error_code.value if res.error else None,
            'message': res.error.message if res.error else None,
            'retryable': res.error.retryable if res.error else None,
            'hint': res.error.hint if res.error else None,
            'retry_after': res.error.retry_after if res.error else None,

            'truncated': res.truncated,
            'bytes_out': len(res.content),
            'duration_ms': ms,
        })

    def error(self, step: int, kind: str, message: str, latency_ms: int, budget: Budget) -> None:
        """
        Record a non-fatal error event.

        ``message`` must be a machine-readable code (type name, reason string, or ``ContextOverflow.code``), never
        ``str(exc)`` — that may contain PII. The only carve-out is ``__exit__`` (unhandled, diagnostic only).
        Log the detail separately.

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

        Args:
            reason (TerminationReason): Why the run ended.
            budget (Budget): Final budget snapshot.
            answer (str | None): Final answer; only its length is stored (``answer_chars``).
            finalization (Finalization): How the answer was produced. Defaults to ``NONE``.
        """
        self._write({
            't': 'finish',
            'reason': reason.value,
            'budget': budget.snapshot(),
            'answer_chars': len(answer or ''),
            'finalization': finalization.value,
        })
