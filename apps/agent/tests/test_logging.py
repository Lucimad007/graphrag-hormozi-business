import json
import logging

from agent.config.logging import JsonFormatter, configure_logging


def test_json_formatter_includes_level_and_message() -> None:
    record = logging.LogRecord(
        name="agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["logger"] == "agent.test"


def test_configure_logging_sets_root_level() -> None:
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    configure_logging("INFO")
