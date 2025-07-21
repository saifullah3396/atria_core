"""
Logging Core Package Initialization

This module initializes the `logging_core` package and exposes the `get_logger`
function for external use. The `get_logger` function provides a centralized
logging utility for the application, ensuring consistent logging behavior
across all modules.

Exports:
    - get_logger: A utility function to retrieve a configured logger instance.

Dependencies:
    - atria_core.logger.logger: Contains the implementation of the `get_logger` function.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from atria_core.logger.logger import LoggerBase, get_logger

__all__ = ["get_logger", "LoggerBase"]
