"""
Logger Filters Module

This module defines custom logging filters for distributed systems. These filters
allow log records to be processed conditionally based on the rank of the current
process, which is useful in distributed training setups.

Classes:
    - DistributedFilter: A logging filter that processes log records only if the rank is zero.

Dependencies:
    - logging: For defining and managing log filters.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import logging

class DistributedFilter(logging.Filter):
    """
    A logging filter that allows log records to be processed only if the rank is zero.

    This filter is useful in distributed systems where only the process with rank zero
    should log messages to avoid duplicate logs from multiple processes.

    Attributes:
        rank (int): The rank of the current process.
    """

    def __init__(self, rank: int) -> None:
        """
        Initializes the DistributedFilter with the given rank.

        Args:
            rank (int): The rank of the current process.
        """
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determines if the log record should be processed based on the rank.

        Args:
            record (logging.LogRecord): The log record to be processed.

        Returns:
            bool: True if the rank is zero, False otherwise.
        """
        return self.rank == 0
