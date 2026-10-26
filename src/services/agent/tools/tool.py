from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel


class ToolErrorCode(StrEnum):
    VALIDATION = 'validation'
    UNKNOWN_TOOL = 'unknown_tool'
    EXECUTION_FAILED = 'execution_failed'
    TEMPORARY_FAILURE = 'temporary_failure'


@dataclass(frozen=True)
class ToolError:
    error_code: ToolErrorCode
    message: str
    hint: str
    retryable: bool
    retry_after: float | None = None

    def __post_init__(self) -> None:
        if len(self.hint) > 200:
            object.__setattr__(self, 'hint', self.hint[:197] + '...')

    def to_model(self) -> str:
        """
        Format error message for LLM.

        Returns:
            str: Error message formatted for LLM.
        """
        text = (
            f'ERROR {self.error_code.value}. '
            f'{self.message} '
            f'Hint: {self.hint}.'
        )

        if self.retry_after is not None:
            text += f' Retry after {self.retry_after:.1f} sec.'

        if self.retryable:
            text += ' Retry is possible.'
        else:
            text += ' Retrying with the same arguments will not help.'

        return text


class ToolStatus(StrEnum):
    OK = 'ok'
    ERROR = 'error'


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    content: str = ''
    error: ToolError | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ToolStatus):
            raise TypeError("'status' must be a ToolStatus.")

        if self.status is ToolStatus.OK and self.error is not None:
            raise ValueError('Successful ToolResult must not contain an error.')

        if self.status is ToolStatus.ERROR and self.error is None:
            raise ValueError('Failed ToolResult must contain an error.')

    @property
    def retryable(self) -> bool:
        return self.error is not None and self.error.retryable

    def to_model(self) -> str:
        """
        Convert the tool result into text that can be sent to the model.

        Returns:
            str: Text formatted for LLM.
        """
        # Successful result: return the content and explicitly mark truncation
        if self.status is ToolStatus.OK:
            tail = '\n[...output truncated...]' if self.truncated else ''
            return self.content + tail

        # An error result must contain a ToolError
        # If it does not, the ToolResult violates its internal contract
        return self.error.to_model()


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., str | Awaitable[str]]
    args_model: type[BaseModel]

    @property
    def parameters(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()
