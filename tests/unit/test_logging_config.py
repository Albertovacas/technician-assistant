import logging

import pytest

from pool_assistant.logging_config import _HANDLER_NAME, configure_logging


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


def _own_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() == _HANDLER_NAME]


def test_sets_requested_level() -> None:
    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG


def test_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")

    assert len(_own_handlers()) == 1


def test_silences_noisy_third_party_loggers() -> None:
    configure_logging("DEBUG")

    assert logging.getLogger("httpx").level == logging.WARNING
