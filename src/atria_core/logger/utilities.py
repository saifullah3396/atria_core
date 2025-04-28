"""
Logging Utilities Module

This module provides utility functions for configuring and managing loggers in the
Atria application. These utilities include enabling colored logging, attaching
file handlers, and adding stream handlers to loggers.

Functions:
    - enable_colored_logging: Enable colored logging for a logger.
    - attach_file_handler: Attach a file handler to a logger.
    - attach_stream_handler: Add a stream handler to a logger.

Dependencies:
    - logging: For creating and managing loggers.
    - coloredlogs: For enabling colored logging with custom styles.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import logging
from typing import Dict, TextIO

import coloredlogs


def enable_colored_logging(
    logger: logging.Logger,
    log_level: int,
    styles: Dict[str, Dict[str, str]],
    log_format: str,
) -> None:
    """
    Enable colored logging for the specified logger.

    This function configures the logger to use colored logs, making it easier to
    distinguish log levels visually. The colors and styles can be customized
    using the `styles` argument.

    Args:
        logger (logging.Logger): The logger to configure.
        log_level (int): The logging level (e.g., `logging.DEBUG`, `logging.INFO`).
        styles (Dict[str, Dict[str, str]]): The color styles for different log levels.
        log_format (str): The format string for log messages.

    Returns:
        None
    """
    coloredlogs.install(
        level=log_level, logger=logger, fmt=log_format, level_styles=styles
    )


def attach_file_handler(
    logger: logging.Logger, log_file_path: str, log_level: int, log_format: str
) -> None:
    """
    Attach a file handler to the specified logger.

    This function adds a `FileHandler` to the logger, enabling log messages to
    be written to a specified file. The log level and format can be customized.

    Args:
        logger (logging.Logger): The logger to configure.
        log_file_path (str): The path to the log file.
        log_level (int): The logging level (e.g., `logging.DEBUG`, `logging.INFO`).
        log_format (str): The format string for log messages.

    Returns:
        None
    """
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)


def attach_stream_handler(
    logger: logging.Logger, log_stream: TextIO, log_level: int, log_format: str
) -> None:
    """
    Add a stream handler to the specified logger.

    This function creates a `StreamHandler` to log messages to a given stream
    (e.g., `sys.stdout` or `sys.stderr`). The handler is configured with the
    specified logging level and message format.

    Args:
        logger (logging.Logger): The logger to which the stream handler will be attached.
        log_stream (TextIO): The stream to which log messages will be written (e.g., `sys.stdout`).
        log_level (int): The logging level (e.g., `logging.DEBUG`, `logging.INFO`).
        log_format (str): The format string for log messages.

    Returns:
        None
    """
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(stream_handler)
