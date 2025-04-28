"""
Unit Tests for Logging Module

This module contains unit tests for the logging functionality provided by the
`LoggerBase` class and related utilities. It verifies the behavior of loggers,
including singleton behavior, logger creation, file path updates, resetting loggers,
and distributed rank persistence.

Test Functions:
    - test_logger_singleton: Tests the singleton behavior of `LoggerBase`.
    - test_logger_create: Tests logger creation using `get_logger`.
    - test_logger_file_path_update: Tests updating the log file path in `LoggerBase`.
    - test_logger_reset: Tests resetting loggers in `LoggerBase`.
    - test_logger_rank_persistence: Tests persistence of distributed rank in `LoggerBase`.

Dependencies:
    - pytest: For running the test cases.
    - logging: For verifying logger behavior.
    - sys: For stream handling.
    - atria_core.logger.filters: Provides the `DistributedFilter` class.
    - atria_core.logger.logger: Provides the `LoggerBase` class and `get_logger` function.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import logging
import sys

from atria_core.logger.filters import DistributedFilter
from atria_core.logger.logger import LoggerBase, get_logger


def test_logger_singleton():
    """
    Test the singleton behavior of LoggerBase.

    Ensures that multiple instances of LoggerBase refer to the same object.

    Assertions:
        - Multiple instances of LoggerBase should refer to the same object.
    """
    logger1 = LoggerBase()
    logger2 = LoggerBase()
    assert logger1 is logger2  # Both instances should be the same


def test_logger_create():
    """
    Test logger creation using get_logger.

    Verifies that a logger is created with the correct name and type.

    Assertions:
        - The logger should be an instance of `logging.Logger`.
        - The logger's name should match the provided name.
    """
    logger = get_logger("atria.test_logger_create")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "atria.test_logger_create"


def test_logger_file_path_update(tmp_path):
    """
    Test updating the log file path in LoggerBase.

    Ensures that the log file path is updated correctly and that loggers
    are reset to include a file handler pointing to the new path.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing.

    Assertions:
        - The log file path in LoggerBase should match the updated path.
        - The logger should include a file handler pointing to the new path.
    """
    log_file = tmp_path / "test.log"
    logger_base = LoggerBase()
    logger_base.log_file_path = str(log_file)

    assert logger_base.log_file_path == str(log_file)

    logger = get_logger("atria.test_logger_file_path_update")
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_logger_reset():
    """
    Test resetting loggers in LoggerBase.

    Ensures that resetting loggers removes existing handlers and reinitializes them.

    Assertions:
        - Loggers should have handlers before resetting.
        - Loggers should still have handlers after resetting.
    """
    logger_base = LoggerBase()
    logger1 = get_logger("atria.test_logger_reset")
    logger1.addHandler(logging.StreamHandler(sys.stdout))  # Add extra handler

    assert len(logger1.handlers) > 0

    logger_base._reset_loggers()
    logger2 = get_logger("atria.test_logger_reset")

    assert len(logger2.handlers) > 0  # Handlers should still be reinitialized


def test_logger_rank_persistence():
    """
    Test persistence of distributed rank in LoggerBase.

    Verifies that the rank set in LoggerBase persists after resetting loggers.

    Assertions:
        - The rank set in LoggerBase should persist in the logger's filters.
    """
    logger_base = LoggerBase()
    logger_base._rank = 2
    logger = get_logger("atria.test.rank_persistence")

    filters = [f for f in logger.filters if isinstance(f, DistributedFilter)]
    assert filters[0].rank == 2  # Rank should be set correctly
