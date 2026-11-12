import json
from datetime import datetime, timezone as dt_timezone
from types import MappingProxyType
from typing import Final, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WEEKDAYS_EN: Final[Mapping[int, str]] = MappingProxyType({
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
})

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=dt_timezone.utc)


def _format_utc_offset(now: datetime) -> str:
    """
    Return the UTC offset of an aware datetime as '+HHMM' / '-HHMM'.

    Contract: `now` must be aware; naive datetimes raise ValueError because silently assuming UTC would hide
    caller bugs. Sub-minute offsets are truncated (never occurs for ZoneInfo tzinfos).

    Raises:
        ValueError: if `now` is naive (has no tzinfo).
    """
    offset = now.utcoffset()
    if offset is None:
        raise ValueError(f"_format_utc_offset requires an aware datetime, got naive: {now!r}")

    total_seconds = int(offset.total_seconds())
    sign = '+' if total_seconds >= 0 else '-'
    total_seconds = abs(total_seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60

    return f'{sign}{hours:02d}{minutes:02d}'


def get_datetime(timezone: Optional[str] = None, *, _now: Optional[datetime] = None) -> str:
    """
    Return the current date and time in the given IANA timezone (UTC by default) as a JSON-encoded string.

    CONTRACT: `timezone` is assumed already validated by GetDatetimeArgs (None or a valid IANA name); unvalidated input
    raises ZoneInfoNotFoundError. Without `tzdata` the default UTC path still works via stdlib `timezone.utc`,
    while non-UTC requests are rejected at validation time. `_now` is test-only and must be aware.

    Args:
        timezone: IANA timezone name, or None for UTC.
        _now: Test-only injection point. If provided, must be an aware datetime; it is used instead of
            `datetime.now(tz)`. Production callers must not pass this.

    Returns:
        str: JSON-encoded object with the following fields:
            - datetime: ISO-8601 timestamp with offset (second precision)
            - timezone: resolved tzinfo name (as `str(now.tzinfo)`; UTC-equivalent aliases are not normalized)
            - utc_offset: numeric offset '+HHMM' / '-HHMM'
            - date: 'YYYY-MM-DD'
            - time: 'HH:MM:SS'
            - day_of_week: English weekday name (Monday..Sunday)
            - iso_weekday: 1 (Monday) .. 7 (Sunday)
            - unix_timestamp: integer epoch seconds (exact, matches `datetime`)
    """
    if timezone:
        tz = ZoneInfo(timezone)
    else:
        try:
            tz = ZoneInfo("UTC")
        except ZoneInfoNotFoundError:
            # `tzdata` is missing, but UTC is representable via the stdlib alone, so the default case still works;
            # non-UTC requests need tzdata and are rejected at validation time with a clear error.
            tz = dt_timezone.utc

    source = _now if _now is not None else datetime.now(tz)
    now = source.replace(microsecond=0)
    # Integer arithmetic: `datetime.timestamp()` returns a float, which is
    # exact for dates near the epoch but can lose a second far from it.
    delta = now - _EPOCH
    unix_timestamp = delta.days * 86400 + delta.seconds
    payload = {
        "datetime": now.isoformat(),
        "timezone": str(now.tzinfo),
        "utc_offset": _format_utc_offset(now),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        # Dictionary lookup, not strftime("%A"): immune to LC_TIME.
        "day_of_week": _WEEKDAYS_EN[now.isoweekday()],
        "iso_weekday": now.isoweekday(),
        "unix_timestamp": unix_timestamp,
    }

    return json.dumps(payload, ensure_ascii=False)
