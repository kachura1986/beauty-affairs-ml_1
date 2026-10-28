import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any

from services.llm import Reply
from .domain import Budget, TerminationReason
from .tools import ToolResult

# How many trace files to keep in RUNS_DIR.
MAX_TRACE_FILES = 500


class Trace:
    """
    A trace records the execution of an agent_graph run, including its input, every LLM call,
    and the final result. The trace is stored in a JSON Lines (JSONL) file, one record per line.

    Lifecycle:
        - ``run_id`` is generated internally (UTC timestamp + random suffix),
          so file names are unique, ordered by time, and cannot escape RUNS_DIR.
        - The trace must be closed via :meth:`close` or by using it as a
          context manager. Calling :meth:`close` more than once is safe.
        - ``__del__`` provides a best-effort fallback so a forgotten handle is still released.
        - Old trace files are pruned on every new trace creation, keeping at most ``MAX_TRACE_FILES`` most recent files.
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

        # Keep runs_dir bounded. Done after opening our own file so that the
        # fresh file participates in the "most recent" ordering
        self._prune_old_runs()

    @staticmethod
    def generate_run_id() -> str:
        """
        Generate a new run identifier with a UTC timestamp prefix.

        Format: ``YYYYMMDD_HHMMSS_<hex>`` — date, then time, then 8 hex chars.
        Safe to use directly as a filename.
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        suffix = secrets.token_hex(4)

        return f'{timestamp}_{suffix}'

    def _prune_old_runs(self) -> None:
        """
        Keep only the most recent ``MAX_TRACE_FILES`` trace files. Best-effort: any failure
        (including permission errors while deleting a stale file) must not affect the current run.
        """
        try:
            files = sorted(self._runs_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return

        for stale in files[MAX_TRACE_FILES:]:
            try:
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

        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> 'Trace':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # On an unexpected exception, try to record it in the trace,
        # but never let the write itself mask the original exception
        if exc_type is not None and not self._closed:
            try:
                self._write({
                    't': 'unexpected_error',
                    'kind': exc_type.__name__,
                    'message': str(exc)[:500],
                })
            except Exception:  # noqa: BLE001
                pass
        self.close()
        # Returning None/False propagates the original exception unchanged

    def __del__(self) -> None:
        # Defensive: if the caller forgot to close the trace,
        # release the file descriptor during GC. Must not raise from __del__
        try:
            if not getattr(self, '_closed', True):
                self.close()
        except Exception:  # noqa: BLE001
            pass

    def _write(self, rec: dict[str, Any]) -> None:
        if self._closed:
            return

        rec = {
            'run_id': self._run_id,
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            **rec,
        }

        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        self._fh.flush()

    @staticmethod
    def _hash(obj: Any) -> str:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)

        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def step(self, n: int, window: list[dict], reply: Reply, budget: Budget, latency_ms: int) -> None:
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
            'decision': 'tool_call' if reply.tool_calls else ('answer' if reply.text.strip() else 'empty'),
            'tools': [c['name'] for c in reply.tool_calls],
            'budget': budget.snapshot(),
        })

    def tool(self, n: int, call: dict, res: ToolResult, ms: int) -> None:
        self._write({
            't': 'tool',
            'step': n,
            'tool': call['name'],
            'args': call.get('arguments', {}),
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
        self._write({
            't': 'error',
            'step': step,
            'kind': kind,
            'message': message,
            'latency_ms': latency_ms,
            'budget': budget.snapshot(),
        })

    def finish(self, reason: TerminationReason, budget: Budget, answer: str | None) -> None:
        self._write({
            't': 'finish',
            'reason': reason.value,
            'budget': budget.snapshot(),
            'answer_chars': len(answer or ''),
        })
