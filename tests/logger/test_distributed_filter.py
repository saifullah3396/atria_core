"""
Unit Tests for DistributedFilter

This module contains unit tests for the `DistributedFilter` class, which is used
to filter log records based on the rank of the current process in distributed systems.

Test Functions:
    - test_distributed_filter_rank_zero: Tests that log records are processed when the rank is zero.
    - test_distributed_filter_nonzero_rank: Tests that log records are blocked when the rank is nonzero.

Dependencies:
    - pytest: For running the test cases.
    - logging: For creating and verifying log records.
    - atria_core.logger.filters: Provides the `DistributedFilter` class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import logging

from atria_core.logger.filters import DistributedFilter


def test_distributed_filter_rank_zero():
    """
    Test DistributedFilter with rank zero.

    Verifies that log records are processed when the rank is zero.

    Assertions:
        - The filter should allow log records to pass when the rank is zero.
    """
    filter = DistributedFilter(rank=0)
    record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", None, None)
    assert filter.filter(record) is True


def test_distributed_filter_nonzero_rank():
    """
    Test DistributedFilter with a nonzero rank.

    Verifies that log records are blocked when the rank is nonzero.

    Assertions:
        - The filter should block log records when the rank is nonzero.
    """
    filter = DistributedFilter(rank=1)
    record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", None, None)
    assert filter.filter(record) is False
