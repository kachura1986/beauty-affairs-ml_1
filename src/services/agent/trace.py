import hashlib
import json
import time
from typing import Any

from services.llm import Reply
from core.config import settings
from .domain import Budget, TerminationReason
from .tools import ToolResult

RUNS_DIR = settings.RUNS_DIR


class Trace:
    """Трейс: JSONL, всегда включён"""

    def __init__(self, run_id: str) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = RUNS_DIR / f'{run_id}.jsonl'
        self._fh = self.path.open('a', encoding='utf-8')

    def _write(self, rec: dict[str, Any]) -> None:
        rec = {'run_id': self.run_id, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **rec}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
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
            'status': res.status,

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
        self._write({'t': 'finish',
                     'reason': reason.value,
                     'budget': budget.snapshot(),
                     'answer_chars': len(answer or '')})
        self._fh.close()
