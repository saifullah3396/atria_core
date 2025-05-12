"""
Centralized Logging Utility

This module provides a centralized logging utility for the Atria application with
support for distributed environments. It includes a singleton-based logger manager
(`LoggerBase`) that ensures consistent logging configuration across the application.

Functions:
    - get_logger: Retrieve a logger configured for distributed environments.
    - set_log_level: Update the log level for all managed loggers.

Classes:
    - LoggerBase: A singleton logger configuration manager for consistent logging.

Dependencies:
    - logging: For creating and managing loggers.
    - os: For accessing environment variables.
    - atria_core.logger.constants: Provides default color styles and root logger name.
    - atria_core.logger.filters: Provides a distributed filter for loggers.
    - atria_core.logger.utilities: Provides utilities for attaching handlers and enabling colored logs.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import logging
import os
import sys
from typing import Optional

from atria_core.logger.constants import _DEFAULT_COLOR_STYLES, _ROOT_LOGGER_NAME
from atria_core.logger.filters import DistributedFilter
from atria_core.logger.utilities import (
    attach_file_handler,
    attach_stream_handler,
    enable_colored_logging,
)


class LoggerBase:
    """
    Singleton logger configuration manager for consistent logging across the application.

    This class ensures that all loggers in the application are configured consistently.
    It supports distributed environments by allowing log filtering based on process rank.

    Attributes:
        _instance (LoggerBase): The singleton instance of the logger manager.
        log_file_path (Optional[str]): The path to the log file.
        rank (int): The rank of the current process in a distributed environment.
        name (str): The root logger name.
        format (str): The log message format.
        log_level (int): The logging level (e.g., `logging.INFO`).
        styles (dict): The color styles for different log levels.
    """

    _instance = None

    def __new__(cls):
        """
        Create or retrieve the singleton instance of LoggerBase.

        Returns:
            LoggerBase: The singleton instance of the logger manager.
        """
        if cls._instance is None:
            cls._instance = super(LoggerBase, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the logger manager with default configurations.

        This method sets up the root logger name, log format, log level, and color styles.
        """
        self._name = _ROOT_LOGGER_NAME
        self._format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        self._log_level = (
            log_level
            if isinstance(log_level, int)
            else logging.getLevelName(log_level.upper())
        )
        self._styles = _DEFAULT_COLOR_STYLES
        self._log_file_path: Optional[str] = None
        self._rank: int = 0

    @property
    def rank(self) -> int:
        """
        Get the current process rank in a distributed environment.

        Returns:
            int: The rank of the current process.
        """
        return self._rank

    @rank.setter
    def rank(self, new_rank: int):
        """
        Update the process rank and reconfigure all loggers.

        Args:
            new_rank (int): The new rank of the current process.
        """
        if new_rank != self._rank:
            self._rank = new_rank
            self._reset_loggers()

    @property
    def log_file_path(self) -> Optional[str]:
        """
        Get the current log file path.

        Returns:
            Optional[str]: The path to the log file, or None if not set.
        """
        return self._log_file_path

    @log_file_path.setter
    def log_file_path(self, log_file_path: str):
        """
        Update the log file path and reconfigure all loggers.

        Args:
            log_file_path (str): The new log file path.
        """
        self._log_file_path = log_file_path
        self._reset_loggers()

    @property
    def log_level(self):
        """
        Get the current logging level.

        Returns:
            int: The current logging level (e.g., `logging.DEBUG`).
        """
        return self._log_level

    @log_level.setter
    def log_level(self, new_level: int):
        """
        Update the logging level and reconfigure all loggers.

        Args:
            new_level (int): The new logging level (e.g., `logging.DEBUG`).
        """
        self._log_level = (
            new_level
            if isinstance(new_level, int)
            else logging.getLevelName(new_level.upper())
        )
        self._reset_loggers()

    def create_logger(
        self, name: Optional[str] = None, reset_logger: bool = False
    ) -> logging.Logger:
        """
        Create or retrieve a logger with the specified name.

        Args:
            name (Optional[str]): The name of the logger. Defaults to the root logger name.
            reset_logger (bool): Whether to reset the logger's handlers. Defaults to False.

        Returns:
            logging.Logger: The configured logger instance.
        """
        name = name or self._name
        logger = logging.getLogger(name)
        if logger.handlers and not reset_logger:
            return logger

        logger.handlers.clear()
        self._configure_handlers(logger)
        logger.propagate = False
        return logger

    def _reset_loggers(self):
        """
        Reset all loggers managed by the logger manager.

        This method clears the handlers of all loggers that start with the root logger name
        and reconfigures them.
        """
        for logger_name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger) and logger_name.startswith(
                ("atria", "atria_core", "__main__")
            ):
                logger.handlers.clear()
                self.create_logger(name=logger_name, reset_logger=True)

    def _configure_handlers(self, logger: logging.Logger):
        """
        Configure the handlers for the specified logger.

        This method attaches a stream handler and, if a log file path is set, a file handler.
        It also enables colored logging.

        Args:
            logger (logging.Logger): The logger to configure.
        """
        logger.addFilter(DistributedFilter(rank=self._rank))
        if self._log_file_path:
            attach_file_handler(
                logger, self._log_file_path, self._log_level, self._format
            )
        attach_stream_handler(
            logger,
            log_stream=sys.stdout,
            log_level=self._log_level,
            log_format=self._format,
        )
        enable_colored_logging(logger, self._log_level, self._styles, self._format)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a logger configured for distributed environments.

    Args:
        name (Optional[str]): The name of the logger. Defaults to the root logger name.

    Returns:
        logging.Logger: The configured logger instance.
    """
    return LoggerBase().create_logger(name=name)


def set_log_level(log_level: int):
    """
    Update the log level for all managed loggers.

    Args:
        log_level (int): The new logging level (e.g., `logging.DEBUG`).
    """
    LoggerBase().log_level = log_level
