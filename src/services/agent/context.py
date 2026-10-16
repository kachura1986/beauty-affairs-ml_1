from typing import Any

from .domain import Task
from .tools import ToolResult

SYSTEM_PROMPT = """Ты инженерный агент. Работай по шагам.

Правила:
1. Если нужно действие - вызывай инструмент, не описывай его словами.
2. Не придумывай результаты инструментов.
3. Результат инструмента — единственный источник фактов о данных.
4. Если инструмент вернул ОШИБКУ, не повторяй тот же вызов с теми же
    аргументами: измени аргументы или выбери другой путь.
5. Когда задача решена, дай финальный ответ без вызова инструментов.
6. Если данных не хватает и получить их нечем — скажи об этом прямо.
"""


def build_window(task: Task, history: list[dict], warning: str | None) -> list[dict]:
    window: list[dict] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    if warning:
        window.append({'role': 'system', 'content': warning})
    window.extend(history)

    return window


def observation(text: str) -> dict[str, str]:
    return {'role': 'user', 'content': f'[наблюдение] {text}'}


def tool_message(call: dict, res: ToolResult) -> dict[str, Any]:
    return {
        'role': 'tool', 'name': call['name'],
        'tool_call_id': call.get('id', call['name']),
        'content': res.to_model()
    }
