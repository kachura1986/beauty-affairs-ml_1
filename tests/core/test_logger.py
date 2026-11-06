import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest
import structlog

from core.logger import (
    _ALLOWED_KEYS,
    _BASE_KEYS,
    _EXTRA_WHITELIST,
    _add_whitelisted_extras,
    _build_shared_processors,
    _filter_event_dict,
    _is_ours,
    get_logger,
    setup_logging,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(level: int = logging.INFO, msg: str = 'msg') -> logging.LogRecord:
    """Construct a minimal LogRecord for direct processor tests."""
    return logging.LogRecord('test', level, __file__, 1, msg, None, None)


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSON-per-line log file into a list of records."""
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_logging():
    """Isolate global logging/structlog state between tests.

    setup_logging early-returns only when *our* handlers (marked with
    _structlog_handler) are already present. This fixture guarantees
    no such handlers exist at the start of each test: root.handlers is
    cleared, structlog is reset, and the get_logger lru_cache is
    dropped. Handlers added during the test are closed and removed in
    the finally block; the original handlers are restored.

    Foreign handlers (e.g. attached by pytest plugins *after* this
    fixture runs) are ignored by setup_logging — it will still install
    ours alongside them.
    """
    from core import logger as logger_mod

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    root.handlers[:] = []
    root.setLevel(logging.WARNING)
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    logger_mod.get_logger.cache_clear()

    try:
        yield root
    finally:
        for h in root.handlers:
            if h not in saved_handlers:
                try:
                    h.close()
                except Exception:
                    pass
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        structlog.reset_defaults()
        structlog.contextvars.clear_contextvars()
        logger_mod.get_logger.cache_clear()


@pytest.fixture
def last_record(tmp_path):
    def _get():
        return _read_jsonl(tmp_path / 'app.log')[-1]

    return _get


# ---------------------------------------------------------------------------
# Constants: the whitelist is the contract
# ---------------------------------------------------------------------------

def test_extra_whitelist_contents():
    """The whitelist contains exactly the canonical extra= keys."""
    assert set(_EXTRA_WHITELIST) == {
        'run_id', 'step', 'tool', 'error_code', 'latency_ms', 'log_file',
    }


def test_extra_whitelist_rejects_legacy_field_names():
    """Regression: 'tool_name' / 'exception_type' must not be in the whitelist.

    services/agent/tools/errors.py used to pass extra={'tool_name': ...,
    'exception_type': ...}; those keys were silently dropped. The fix
    renamed them to 'tool' / 'error_code'. This test pins that the old
    names are not accidentally re-added.
    """
    assert 'tool_name' not in _EXTRA_WHITELIST
    assert 'exception_type' not in _EXTRA_WHITELIST


def test_base_keys_contents():
    """Base JSON event fields are never filtered out."""
    assert set(_BASE_KEYS) == {
        'timestamp', 'level', 'logger', 'event', 'exception', 'stack',
    }


def test_allowed_keys_is_union_of_base_and_whitelist():
    assert _ALLOWED_KEYS == _BASE_KEYS | frozenset(_EXTRA_WHITELIST)


# ---------------------------------------------------------------------------
# _add_whitelisted_extras
# ---------------------------------------------------------------------------

def test_add_whitelisted_extras_copies_whitelisted_fields():
    """All whitelisted extras from LogRecord.__dict__ reach event_dict."""
    record = _record()
    record.run_id = 'r1'
    record.step = 5
    record.tool = 'search'
    record.error_code = 'VALIDATION'
    record.latency_ms = 42
    record.log_file = '/tmp/x.log'

    event_dict = {'event': 'msg', '_record': record}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result['run_id'] == 'r1'
    assert result['step'] == 5
    assert result['tool'] == 'search'
    assert result['error_code'] == 'VALIDATION'
    assert result['latency_ms'] == 42
    assert result['log_file'] == '/tmp/x.log'


def test_add_whitelisted_extras_copies_only_whitelisted_fields():
    """Positive control: whitelisted fields are copied; others are not.

    Without a positive control (a whitelisted key that DOES land in
    event_dict), this test would pass even if the processor returned
    event_dict unchanged.
    """
    record = _record()
    record.run_id = 'r1'  # whitelisted — must be copied
    record.arbitrary = 'x'  # arbitrary — must be dropped
    record.tool_name = 'search'  # legacy — must be dropped
    record.exception_type = 'ValueError'  # legacy — must be dropped

    event_dict = {'event': 'msg', '_record': record}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result['run_id'] == 'r1'
    assert 'arbitrary' not in result
    assert 'tool_name' not in result
    assert 'exception_type' not in result


def test_add_whitelisted_extras_skips_none_values():
    """Whitelisted attributes set to None are not copied."""
    record = _record()
    record.run_id = None
    record.step = None

    event_dict = {'event': 'msg', '_record': record}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert 'run_id' not in result
    assert 'step' not in result


def test_add_whitelisted_extras_keeps_falsy_non_none():
    """Regression: 0 and '' are valid whitelisted values, not skipped.

    The processor uses `if value is not None`, so 0 / 0.0 / '' must be kept.
    A refactor to `if value:` would drop step=0, latency_ms=0 and
    error_code='' without any other test failing.
    """
    record = _record()
    record.step = 0
    record.latency_ms = 0
    record.error_code = ''
    record.run_id = ''

    event_dict = {'event': 'msg', '_record': record}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result['step'] == 0
    assert result['latency_ms'] == 0
    assert result['error_code'] == ''
    assert result['run_id'] == ''


def test_add_whitelisted_extras_skips_missing_attributes():
    """Whitelisted keys absent from the record are not added.

    The *missing* path is isolated by guarding that the helper's
    default LogRecord does not already carry any whitelisted key
    (even as None). Without this guard, a helper change that added
    e.g. ``record.tool = None`` would make the test silently exercise
    the *None* path instead of the *missing* path — the None case is
    covered separately by ``test_add_whitelisted_extras_skips_none_values``.
    """
    record = _record()
    record.run_id = 'r1'
    for key in ('step', 'tool', 'error_code', 'latency_ms', 'log_file'):
        assert key not in record.__dict__  # guard against helper changes

    event_dict = {'event': 'msg', '_record': record}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result['run_id'] == 'r1'
    for missing in ('step', 'tool', 'error_code', 'latency_ms', 'log_file'):
        assert missing not in result


def test_add_whitelisted_extras_no_record_is_noop():
    """Native structlog calls have no `_record`; the processor is a no-op."""
    event_dict = {'event': 'msg', 'arbitrary': 'x'}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result is event_dict
    assert result['arbitrary'] == 'x'


def test_add_whitelisted_extras_returns_same_dict():
    """The processor returns the same event_dict instance (structlog contract).

    Note: the current implementation also mutates in-place, but that is
    an implementation detail, NOT part of the contract. A future pure
    refactor (like _filter_event_dict) must not break this test.
    """
    record = _record()
    record.run_id = 'r1'

    event_dict = {'event': 'msg', '_record': record}
    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result is event_dict
    assert event_dict['run_id'] == 'r1'  # current behavior, not contract


def test_add_whitelisted_extras_overwrites_existing_value():
    """A whitelisted key already in event_dict is overwritten by the record.

    Pinned for clarity: in practice native calls have no `_record`, and
    stdlib calls never put whitelisted keys into event_dict before this
    processor runs, so this overwrite is theoretical.
    """
    record = _record()
    record.run_id = 'from-record'
    event_dict = {'event': 'msg', 'run_id': 'from-event-dict', '_record': record}

    result = _add_whitelisted_extras(None, 'info', event_dict)

    assert result['run_id'] == 'from-record'


# ---------------------------------------------------------------------------
# _filter_event_dict
# ---------------------------------------------------------------------------

def test_filter_event_dict_keeps_base_keys():
    """All base fields survive the filter."""
    result = _filter_event_dict(None, 'info', {
        'timestamp': 'ts', 'level': 'info', 'logger': 'l',
        'event': 'msg', 'exception': 'exc', 'stack': 'st',
    })
    assert set(result) == _BASE_KEYS


def test_filter_event_dict_keeps_whitelisted_extras():
    """All whitelisted extras survive the filter."""
    result = _filter_event_dict(None, 'info', {
        'event': 'x', 'run_id': 'r1', 'tool': 't', 'error_code': 'E',
    })
    assert result['run_id'] == 'r1'
    assert result['tool'] == 't'
    assert result['error_code'] == 'E'


def test_filter_event_dict_keeps_falsy_whitelisted_values():
    """Regression: 0 / '' whitelisted values must not be dropped.

    Only None is dropped; falsy-but-not-None values are preserved. A
    refactor to a truthiness check would drop step=0, latency_ms=0,
    error_code='', run_id=''.

    The positive control (tool='search') guards against a filter that
    drops everything: without it, the test would not distinguish a
    correct filter from `return {}`.
    """
    result = _filter_event_dict(None, 'info', {
        'event': 'x',
        'step': 0,
        'latency_ms': 0,
        'error_code': '',
        'run_id': '',
        'tool': 'search',  # positive control
    })
    assert result['step'] == 0
    assert result['latency_ms'] == 0
    assert result['error_code'] == ''
    assert result['run_id'] == ''
    assert result['tool'] == 'search'


def test_filter_event_dict_drops_arbitrary_keys():
    """Keys outside the base set and whitelist are dropped."""
    result = _filter_event_dict(None, 'info', {
        'event': 'x', 'arbitrary': 'y', 'tool_name': 'z',
    })
    assert set(result) == {'event'}


def test_filter_event_dict_drops_structlog_meta_keys():
    """structlog internal markers do not leak into JSON."""
    result = _filter_event_dict(None, 'info', {
        'event': 'x', '_record': object(), '_from_structlog': True,
    })
    assert '_record' not in result
    assert '_from_structlog' not in result


def test_filter_event_dict_drops_whitelisted_none_values():
    """Whitelisted extras with a None value are dropped.

    The positive control (step=3) guards against a filter that drops
    everything: without it, the test would not distinguish a correct
    filter from `return {}`.
    """
    result = _filter_event_dict(None, 'info', {
        'event': 'x', 'run_id': None, 'tool': None, 'error_code': None,
        'step': 3,  # positive control
    })
    assert 'run_id' not in result
    assert 'tool' not in result
    assert 'error_code' not in result
    assert result['step'] == 3


def test_filter_event_dict_keeps_base_keys_with_none():
    """Base keys with None are kept; only whitelisted extras are None-dropped."""
    result = _filter_event_dict(None, 'info', {
        'event': 'x', 'exception': None,
    })
    assert 'exception' in result


def test_filter_event_dict_drops_internal_logrecord_fields():
    """Internal LogRecord attributes never leak into JSON.

    Guards against a future refactor that would pass the raw record through
    without going via _add_whitelisted_extras (which only copies whitelisted
    attributes) — the filter is the last line of defence.
    """
    event_dict = {
        'event': 'x',
        'msg': 'x', 'args': (), 'exc_info': None, 'levelname': 'INFO',
        'levelno': 20, 'stack_info': None, 'lineno': 1,
        'pathname': '/p', 'filename': 'f.py', 'funcName': 'f',
        'name': 'n', 'msecs': 1.0, 'thread': 1, 'threadName': 't',
        'process': 1, 'processName': 'p',
    }
    result = _filter_event_dict(None, 'info', event_dict)
    assert set(result) == {'event'}


def test_filter_event_dict_does_not_mutate_input():
    """The filter is pure: it must not modify its argument."""
    original = {'event': 'x', 'run_id': 'r1', 'arbitrary': 1, 'msg': 'm'}
    snapshot = dict(original)

    _filter_event_dict(None, 'info', original)

    assert original == snapshot


def test_filter_event_dict_returns_new_dict():
    """A fresh dict is returned even when the input contains only allowed keys."""
    original = {'event': 'x'}
    result = _filter_event_dict(None, 'info', original)
    assert result is not original


def test_filter_event_dict_empty_input():
    assert _filter_event_dict(None, 'info', {}) == {}


# ---------------------------------------------------------------------------
# _build_shared_processors
# ---------------------------------------------------------------------------

def test_build_shared_processors_returns_list_of_callables():
    procs = _build_shared_processors()
    assert isinstance(procs, list)
    assert procs
    assert all(callable(p) for p in procs)


def test_build_shared_processors_includes_whitelist_processor():
    """The whitelist-enrichment step is part of the foreign-log pipeline."""
    assert _add_whitelisted_extras in _build_shared_processors()


def test_build_shared_processors_excludes_json_renderer():
    """The JSONRenderer must live only inside ProcessorFormatter.processors.

    Including it here would double-encode every foreign (stdlib) log record:
    the processor would produce a JSON string, then the formatter would
    serialise that string again.
    """
    for proc in _build_shared_processors():
        assert not isinstance(proc, structlog.processors.JSONRenderer)


# ---------------------------------------------------------------------------
# _is_ours
# ---------------------------------------------------------------------------

def test_is_ours_true_for_marked_handler():
    h = logging.NullHandler()
    h._structlog_handler = True
    assert _is_ours(h) is True


def test_is_ours_false_for_unmarked_handler():
    assert _is_ours(logging.NullHandler()) is False


def test_is_ours_false_for_false_marker():
    h = logging.NullHandler()
    h._structlog_handler = False
    assert _is_ours(h) is False


# ---------------------------------------------------------------------------
# setup_logging: filesystem and handlers
# ---------------------------------------------------------------------------

def test_setup_logging_creates_log_file(tmp_path):
    setup_logging(tmp_path)
    assert (tmp_path / 'app.log').exists()


def test_setup_logging_creates_nested_log_directory(tmp_path):
    """A missing parent chain is created recursively."""
    target = tmp_path / 'nested' / 'deep' / 'logs'
    assert not target.exists()

    setup_logging(target)

    assert target.exists()


def test_setup_logging_returns_root_logger(tmp_path):
    logger = setup_logging(tmp_path)
    assert logger is logging.getLogger()


def test_setup_logging_sets_level_to_info(tmp_path):
    setup_logging(tmp_path)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_adds_one_file_and_one_console_handler(tmp_path):
    """Exactly one file handler and one plain console handler are attached."""
    setup_logging(tmp_path)
    handlers = logging.getLogger().handlers

    ours = [h for h in handlers if _is_ours(h)]
    files = [h for h in ours if isinstance(h, TimedRotatingFileHandler)]
    consoles = [
        h for h in ours
        if type(h) is logging.StreamHandler  # exact type: not a FileHandler subclass
    ]

    assert len(files) == 1
    assert len(consoles) == 1


def test_setup_logging_uses_midnight_rotation(tmp_path):
    """Rotation policy: midnight, daily, 7 backups, UTF-8.

    TimedRotatingFileHandler normalizes `interval` to seconds at init
    when `when` is a date-based unit, so with when='midnight' and
    interval=1 the attribute is 86400 (one day), not 1.
    """
    setup_logging(tmp_path)
    (handler,) = [
        h for h in logging.getLogger().handlers
        if _is_ours(h) and isinstance(h, TimedRotatingFileHandler)
    ]
    assert handler.when == 'MIDNIGHT'
    assert handler.interval == 86400  # one day, in seconds
    assert handler.backupCount == 7
    assert handler.encoding == 'utf-8'


def test_setup_logging_is_idempotent(tmp_path):
    """A second call does not add duplicate handlers.

    Counts only our handlers (marked with _structlog_handler): foreign
    handlers added by other plugins must not affect this assertion.
    """
    logger1 = setup_logging(tmp_path)
    n_ours = len([h for h in logging.getLogger().handlers if _is_ours(h)])

    logger2 = setup_logging(tmp_path)

    assert logger1 is logger2
    assert len([h for h in logging.getLogger().handlers if _is_ours(h)]) == n_ours


def test_setup_logging_does_not_switch_dir_on_second_call(tmp_path):
    """Idempotency contract: a second call with a different dir does nothing.

    Documents current behavior: handlers are already installed, so logs
    continue to go to the first directory. A future "reinit" refactor
    must change this test deliberately.
    """
    first = tmp_path / 'a'
    second = tmp_path / 'b'

    setup_logging(first)
    setup_logging(second)

    assert (first / 'app.log').exists()
    assert not (second / 'app.log').exists()


def test_setup_logging_writes_diagnostic_record(tmp_path):
    """setup_logging emits a diagnostic record into the configured file."""
    setup_logging(tmp_path)
    records = _read_jsonl(tmp_path / 'app.log')
    assert any(r['event'].startswith('Logging setup complete') for r in records)


@pytest.mark.parametrize('name', [
    'azure.core.pipeline.policies.http_logging_policy',
    'httpx',
    'httpcore',
])
def test_setup_logging_quiets_noisy_loggers(tmp_path, name):
    """Verbose HTTP client loggers are pinned to WARNING."""
    setup_logging(tmp_path)
    assert logging.getLogger(name).level == logging.WARNING


# ---------------------------------------------------------------------------
# setup_logging: JSON output shape
# ---------------------------------------------------------------------------

def test_output_is_one_json_object_per_line(tmp_path):
    """Every line of the log file is a standalone JSON object.

    The test does not pin an exact line count: setup_logging itself writes
    a diagnostic record ('Logging setup complete'), and that count is an
    implementation detail, not a contract. Instead it verifies that every
    line parses as a JSON object and that the caller's records are the
    last two events.
    """
    logger = setup_logging(tmp_path)
    logger.info('first')
    logger.info('second')

    lines = (tmp_path / 'app.log').read_text(encoding='utf-8').splitlines()
    for line in lines:
        assert isinstance(json.loads(line), dict)

    events = [json.loads(line)['event'] for line in lines]
    assert events[-2:] == ['first', 'second']


def test_no_double_encoding(tmp_path):
    """Each line parses as a dict, not as a JSON-encoded string.

    Double encoding is the symptom of JSONRenderer being applied twice.
    """
    logger = setup_logging(tmp_path)
    logger.info('no-double')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert isinstance(last, dict)
    assert last['event'] == 'no-double'


def test_base_fields_present(tmp_path):
    """A plain log call emits the base fields with correct values.

    setup_logging returns the root logger, so calls made through it are
    attributed to logger='root' — not 'core.logger'. The module name
    only appears if a caller obtains logging.getLogger('core.logger')
    directly, which this test intentionally does not do.
    """
    logger = setup_logging(tmp_path)
    logger.info('with-base-fields')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'timestamp' in last
    assert last['level'] == 'info'
    assert last['logger'] == 'root'
    assert last['event'] == 'with-base-fields'


def test_timestamp_is_iso_utc(tmp_path):
    """The TimeStamper is configured as ISO-8601 UTC."""
    logger = setup_logging(tmp_path)
    logger.info('ts-check')

    ts = _read_jsonl(tmp_path / 'app.log')[-1]['timestamp']
    assert 'T' in ts
    assert ts.endswith('Z') or ts.endswith('+00:00')


def test_whitelisted_extras_land_in_json(tmp_path):
    """Every whitelisted key passed through extra= is visible in JSON."""
    logger = setup_logging(tmp_path)
    logger.info(
        'with-extras',
        extra={
            'run_id': 'r1',
            'step': 3,
            'tool': 'search_products',
            'error_code': 'VALIDATION',
            'latency_ms': 42,
            'log_file': '/tmp/custom.log',
        },
    )

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert last['run_id'] == 'r1'
    assert last['step'] == 3
    assert last['tool'] == 'search_products'
    assert last['error_code'] == 'VALIDATION'
    assert last['latency_ms'] == 42
    assert last['log_file'] == '/tmp/custom.log'


def test_non_whitelisted_extras_dropped(tmp_path):
    """Arbitrary extra= keys are silently filtered out."""
    logger = setup_logging(tmp_path)
    logger.info('with-noise', extra={'custom': 'x', 'another': 42})

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'custom' not in last
    assert 'another' not in last


def test_whitelisted_none_extra_dropped(tmp_path):
    """Whitelisted extras explicitly set to None are dropped from JSON.

    The positive control (tool='search') guards against a filter that
    drops everything: without it, the test would not distinguish a
    correct filter from one that always drops whitelisted extras.
    """
    logger = setup_logging(tmp_path)
    logger.info('none-extra', extra={'run_id': None, 'step': None, 'tool': 'search'})

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'run_id' not in last
    assert 'step' not in last
    assert last['tool'] == 'search'


def test_logrecord_internal_fields_do_not_leak(tmp_path):
    """No raw LogRecord attribute appears in the JSON output."""
    logger = setup_logging(tmp_path)
    logger.info('leak-check')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    for attr in (
            'name', 'levelno', 'pathname', 'filename', 'lineno',
            'funcName', 'msecs', 'thread', 'threadName', 'process',
            'processName', 'msg', 'args', 'levelname',
    ):
        assert attr not in last


def test_unicode_preserved(tmp_path):
    """ensure_ascii=False: non-ASCII characters are written verbatim."""
    logger = setup_logging(tmp_path)
    logger.info('привет', extra={'run_id': 'r-тест'})

    text = (tmp_path / 'app.log').read_text(encoding='utf-8')
    assert 'привет' in text
    assert 'r-тест' in text
    assert '\\u' not in text


def test_exception_rendered_under_exception_key(tmp_path):
    """exc_info is rendered by format_exc_info into the 'exception' field."""
    logger = setup_logging(tmp_path)
    try:
        raise ValueError('boom')
    except ValueError:
        logger.exception('caught')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'exception' in last
    assert 'Traceback' in last['exception']
    assert 'ValueError: boom' in last['exception']


def test_stack_info_rendered_under_stack_key(tmp_path):
    """stack_info=True is rendered by StackInfoRenderer into the 'stack' field.

    This covers the *only* supported way of populating 'stack':
    extra={'stack': ...} is filtered out (not whitelisted), so
    StackInfoRenderer is the canonical path.
    """
    logger = setup_logging(tmp_path)
    logger.info('with-stack', stack_info=True)

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'stack' in last
    assert 'Stack' in last['stack']


def test_extra_stack_is_dropped(tmp_path):
    """Regression: extra={'stack': ...} was silently lost.

    'stack' is in _BASE_KEYS, so _filter_event_dict would pass it through
    unchanged. It is dropped earlier — by _add_whitelisted_extras, which
    only copies keys from _EXTRA_WHITELIST and 'stack' is not in that set.
    Hence the drop is a consequence of the earlier stage, not the filter.
    """
    logger = setup_logging(tmp_path)
    logger.info('legacy-stack', extra={'stack': ['frame1', 'frame2']})

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'stack' not in last


def test_empty_message_is_logged(tmp_path):
    """logger.info('') produces {"event": ""}, not a crash."""
    logger = setup_logging(tmp_path)
    logger.info('')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert last['event'] == ''


def test_exc_info_without_active_exception(tmp_path):
    """exc_info=True without an active exception emits NO exception field.

    structlog's format_exc_info drops (None, None, None) — unlike stdlib's
    Formatter, which would emit 'NoneType: None'. This pins the observable
    behaviour of the JSON pipeline: no misleading 'NoneType: None' string
    ends up in production logs when exc_info is passed without an active
    exception.
    """
    logger = setup_logging(tmp_path)
    logger.error('no-active-exc', exc_info=True)

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'exception' not in last


def test_stack_info_and_exc_info_together(tmp_path):
    """stack_info=True and exc_info coexist: both 'stack' and 'exception' appear."""
    logger = setup_logging(tmp_path)
    try:
        raise ValueError('with-stack-and-exc')
    except ValueError as exc:
        logger.error(
            'both-fields',
            exc_info=(type(exc), exc, exc.__traceback__),
            stack_info=True,
        )

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'stack' in last
    assert 'exception' in last
    assert 'ValueError: with-stack-and-exc' in last['exception']


def test_console_handler_writes_to_stdout(tmp_path, capsys):
    """The console handler mirrors JSON records to stdout."""
    logger = setup_logging(tmp_path)
    logger.info('to-console')

    captured = capsys.readouterr()
    assert 'to-console' in captured.out
    # Every stdout line must still be valid JSON.
    for line in captured.out.splitlines():
        if line.strip():
            json.loads(line)


# ---------------------------------------------------------------------------
# setup_logging: integration with the errors.py pattern
# ---------------------------------------------------------------------------

def test_errors_py_pattern_lands_in_json(tmp_path):
    """Canonical shape from services/agent/tools/errors.py::log_exception.

    After the fix, log_exception passes exc_info and extra={'tool': ...,
    'error_code': ...}. Both must be visible in JSON, at ERROR level.
    """
    logger = setup_logging(tmp_path)
    try:
        raise RuntimeError('tool boom')
    except RuntimeError as exc:
        logger.error(
            'Tool execution failed.',
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={'tool': 'search_products', 'error_code': 'execution_failed'},
        )

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert last['event'] == 'Tool execution failed.'
    assert last['level'] == 'error'
    assert last['tool'] == 'search_products'
    assert last['error_code'] == 'execution_failed'
    assert 'RuntimeError: tool boom' in last['exception']


def test_legacy_errors_py_fields_are_dropped(tmp_path):
    """Regression: the old 'tool_name' / 'exception_type' extras were silently lost.

    The fix renamed them to 'tool' / 'error_code'. This test pins the
    observable behaviour: the old names still do not appear in JSON,
    but the exception type remains visible via the 'exception' field
    (which format_exc_info populates from exc_info).
    """
    logger = setup_logging(tmp_path)
    try:
        raise RuntimeError('boom')
    except RuntimeError as exc:
        logger.error(
            'legacy',
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={'tool_name': 'search', 'exception_type': 'RuntimeError'},
        )

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert 'tool_name' not in last
    assert 'exception_type' not in last
    assert 'RuntimeError' in last['exception']


# ---------------------------------------------------------------------------
# setup_logging: native structlog calls
# ---------------------------------------------------------------------------

def test_native_structlog_call_routes_through_pipeline(tmp_path):
    """Whitelisted kwargs from a native structlog call land in the JSON file.

    This exercises the pipeline end-to-end: structlog.configure was applied
    in setup_logging, the event reached ProcessorFormatter, and whitelisted
    kwargs survived _filter_event_dict. Which individual processor handled
    each field is covered by the dedicated processor tests above; this test
    only pins the observable end-to-end contract.
    """
    setup_logging(tmp_path)
    slog = structlog.get_logger('native.test')
    slog.info('native-event', run_id='r1', tool='search')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert last['event'] == 'native-event'
    assert last['logger'] == 'native.test'
    assert last['run_id'] == 'r1'
    assert last['tool'] == 'search'


def test_native_structlog_drops_non_whitelisted_kwargs(tmp_path):
    """Native calls are filtered by _filter_event_dict too.

    The positive control (run_id='r1') guards against a filter that
    drops everything: without it, the test would not distinguish a
    correct filter from `return {}`.
    """
    setup_logging(tmp_path)
    slog = structlog.get_logger('native')
    slog.info('native-event', run_id='r1', arbitrary='x', tool_name='legacy')

    last = _read_jsonl(tmp_path / 'app.log')[-1]
    assert last['run_id'] == 'r1'
    assert 'arbitrary' not in last
    assert 'tool_name' not in last


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

def test_get_logger_uses_settings_logs_dir(tmp_path, monkeypatch):
    """get_logger writes to settings.LOGS_DIR, not a hard-coded path."""
    from core import logger as logger_mod
    monkeypatch.setattr(logger_mod.settings, 'LOGS_DIR', tmp_path)

    logger_mod.get_logger.cache_clear()
    logger_mod.get_logger()

    assert (tmp_path / 'app.log').exists()


def test_get_logger_caches_setup(tmp_path, monkeypatch):
    """setup_logging is invoked exactly once across repeated get_logger() calls."""
    from core import logger as logger_mod

    calls: list[tuple] = []
    original = logger_mod.setup_logging

    def spy(logs_dir, log_filename='app.log'):
        calls.append((logs_dir, log_filename))
        return original(logs_dir, log_filename)

    monkeypatch.setattr(logger_mod.settings, 'LOGS_DIR', tmp_path)
    monkeypatch.setattr(logger_mod, 'setup_logging', spy)

    logger_mod.get_logger.cache_clear()

    a = logger_mod.get_logger()
    b = logger_mod.get_logger()
    c = logger_mod.get_logger()

    assert a is b is c
    assert len(calls) == 1
    assert calls[0] == (tmp_path, 'app.log')


def test_get_logger_cache_clear_calls_setup_again(tmp_path, monkeypatch):
    """cache_clear() forces setup_logging to be called again on the next call.

    setup_logging itself is idempotent (early-returns when our handlers
    exist), so the second invocation does NOT create new handlers. The
    extra assert on the number of our handlers pins that idempotency —
    otherwise this test would only exercise the lru_cache, not the
    early-return path.
    """
    from core import logger as logger_mod

    calls: list[int] = []
    original = logger_mod.setup_logging

    def spy(logs_dir, log_filename='app.log'):
        calls.append(1)
        return original(logs_dir, log_filename)

    monkeypatch.setattr(logger_mod.settings, 'LOGS_DIR', tmp_path)
    monkeypatch.setattr(logger_mod, 'setup_logging', spy)

    logger_mod.get_logger.cache_clear()
    logger_mod.get_logger()
    logger_mod.get_logger()
    logger_mod.get_logger.cache_clear()

    logger_mod.get_logger()

    # Two cache fills; the middle two calls hit the cached value.
    assert len(calls) == 2
    # setup_logging is idempotent: exactly two of our handlers exist.
    root = logging.getLogger()
    assert len([h for h in root.handlers if _is_ours(h)]) == 2


# ---------------------------------------------------------------------------
# setup_logging: coexistence with foreign handlers
# ---------------------------------------------------------------------------

def test_setup_logging_coexists_with_foreign_handlers(tmp_path):
    """Foreign handlers on root are preserved; ours are appended.

    Regression for the previous behaviour where any handler on root made
    setup_logging early-return, leaving the app with no file handler at all.
    """
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        logger = setup_logging(tmp_path)

        assert logger is logging.getLogger()
        assert (tmp_path / 'app.log').exists()

        ours = [h for h in root.handlers if _is_ours(h)]
        files = [h for h in ours if isinstance(h, TimedRotatingFileHandler)]
        consoles = [h for h in ours if type(h) is logging.StreamHandler]
        assert len(files) == 1
        assert len(consoles) == 1

        assert foreign in root.handlers
        assert root.level == logging.INFO
    finally:
        root.removeHandler(foreign)


def test_setup_logging_is_idempotent_with_foreign_handlers(tmp_path):
    """Second call adds nothing new even when a foreign handler is present."""
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        setup_logging(tmp_path)
        ours_after_first = [h for h in root.handlers if _is_ours(h)]

        setup_logging(tmp_path)
        ours_after_second = [h for h in root.handlers if _is_ours(h)]

        assert len(ours_after_first) == len(ours_after_second) == 2
        assert sum(1 for h in ours_after_second if isinstance(h, TimedRotatingFileHandler)) == 1
        assert sum(1 for h in ours_after_second if type(h) is logging.StreamHandler) == 1
        assert [h for h in root.handlers if h is foreign] == [foreign]
    finally:
        root.removeHandler(foreign)
