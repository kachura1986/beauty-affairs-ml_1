from typing import Final, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.agent.tools import Tool
from services.datetime_utils import get_datetime

_MAX_OUTPUT_CHARS_HARD_CAP: Final[int] = 1000


class GetDatetimeArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    timezone: Optional[str] = Field(
        default=None,
        description=(
            "IANA timezone name, e.g. 'Europe/Moscow', 'America/New_York', 'Asia/Tokyo'.  If omitted or null,"
            "UTC is used. Empty string is rejected: omit the field instead when UTC is intended."
        ),
    )

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate that ``timezone`` is a known IANA name.

        CONTRACT: empty string is rejected explicitly; ``None`` passes through. Distinguishes host misconfiguration
        (no tzdata; the model cannot fix it) from an unknown zone name (model error; retry is possible).

        Args:
            v (Optional[str]): Raw timezone value from the tool call.

        Returns:
            Optional[str]: The validated timezone name, or ``None`` when omitted.

        Raises:
            ValueError: If ``v`` is an empty string, an unknown IANA name, or the
                server has no IANA timezone database installed.
        """
        if v is None:
            return None

        if v == '':
            raise ValueError(
                "timezone must be a valid IANA name or omitted; empty string is not accepted (omit the field for UTC)."
            )

        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            # Distinguish host misconfiguration (no tzdata; model cannot fix it) from
            # an unknown zone name (model error; retry is possible).
            try:
                ZoneInfo('UTC')
            except ZoneInfoNotFoundError:
                raise ValueError(
                    "Server has no IANA timezone database (`tzdata` is not installed). "
                    "Omit `timezone` to get UTC, or report this to the operator."
                ) from exc

            raise ValueError(
                f"Unknown IANA timezone: {v!r}. Use a name like 'Europe/Moscow' or omit the field for UTC."
            ) from exc

        except ValueError:
            raise ValueError(
                f"Unknown IANA timezone: {v!r}. Use a name like 'Europe/Moscow' or omit the field for UTC."
            )

        return v


def create_base_tools(max_output_chars: int) -> list[Tool]:
    """Return the always-registered base tools as a list.

    CONTRACT: callers merge these with domain tools and pass the combined list to ToolRegistry, which validates
    async-callable and unique names in one place. ``default_timeout_seconds`` is applied by ToolRegistry when a
    Tool does not declare ``timeout_seconds``.

    Args:
        max_output_chars (int): Per-tool output character budget; clamped to max_chars for the ``get_datetime`` tool.

    Returns:
        list[Tool]: The base tools, ready to be merged with domain tools.
    """

    async def get_datetime_tool(timezone: Optional[str] = None) -> str:
        """
        Return the current date and time as a JSON string.

        Args:
            timezone (Optional[str]): IANA timezone name; ``None`` means UTC.

        Returns:
            str: JSON payload produced by :func:`services.datetime_utils.get_datetime`.
        """
        return get_datetime(timezone)

    return [
        Tool(
            name="get_datetime",
            description="""
Get the current date and time in an IANA timezone (UTC if omitted).

Call whenever the answer depends on the current moment ("what time is it?", "today's date", "what day is it in Tokyo?",
scheduling relative to "now", etc.). The model's internal sense of the current time is unreliable.

Returns JSON: datetime (ISO-8601, seconds), timezone, utc_offset ("+0300"), date ("YYYY-MM-DD"), time ("HH:MM:SS"),
day_of_week (Monday..Sunday), iso_weekday (1..7), unix_timestamp (epoch seconds).
""",
            fn=get_datetime_tool,
            args_model=GetDatetimeArgs,
            max_output_chars=min(max_output_chars, _MAX_OUTPUT_CHARS_HARD_CAP),
        ),
    ]
