from typing import Final, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.agent.tools import Tool, ToolRegistry
from services.datetime_utils import get_datetime

_MAX_OUTPUT_CHARS_HARD_CAP: Final[int] = 2000


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


def create_base_tools(default_timeout_seconds: float, max_output_chars: int) -> ToolRegistry:
    async def get_datetime_tool(timezone: Optional[str] = None) -> str:
        return get_datetime(timezone)

    return ToolRegistry([
        Tool(
            name="get_datetime",
            description="""
Get the current date and time in a given IANA timezone (UTC if not specified).

Use this tool whenever the answer depends on the current moment: "what time is it?", "what is today's date?",
"what day of the week is it in Tokyo?", scheduling relative to "now", etc. The model's internal knowledge of
the current time is unreliable — always call this tool when freshness matters.

The result is a JSON object with the following fields:
- datetime: ISO-8601 timestamp with offset (second precision)
- timezone: resolved tzinfo name as reported by zoneinfo ("UTC" for omitted or
  literally "UTC"; other UTC aliases like "Etc/UTC" are not normalized)
- utc_offset: numeric offset, e.g. "+0300"
- date: "YYYY-MM-DD"
- time: "HH:MM:SS"
- day_of_week: English weekday name (Monday..Sunday), independent of server locale
- iso_weekday: 1 (Monday) .. 7 (Sunday)
- unix_timestamp: integer epoch seconds (matches `datetime` exactly)

`datetime` and `unix_timestamp` describe the same instant.

Do not pass an empty string for `timezone`; omit the argument to get UTC.

Examples:
- "What time is it?" -> get_datetime()
- "What is today's date in Moscow?" -> get_datetime(timezone="Europe/Moscow")
- "What day is it in New York?" -> get_datetime(timezone="America/New_York")
""",
            fn=get_datetime_tool,
            args_model=GetDatetimeArgs,
            max_output_chars=min(max_output_chars, _MAX_OUTPUT_CHARS_HARD_CAP),
        ),
    ],
        default_timeout_seconds=default_timeout_seconds,
        max_output_chars=max_output_chars,
    )
