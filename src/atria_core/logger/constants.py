"""
Logger Constants Module

This module defines constants used for logging throughout the application. These
constants are used to configure and manage the logging system, ensuring consistency
in log messages and logger names.

Constants:
    - _ROOT_LOGGER_NAME (str): Specifies the root logger name for the application.
    - _DEFAULT_COLOR_STYLES (dict): Contains color configurations for various logging levels.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

# The name of the root logger for the application
_ROOT_LOGGER_NAME = "atria"

# Default color styles for different logging levels
_DEFAULT_COLOR_STYLES = {
    "critical": {"bold": True, "color": "red"},
    "debug": {"color": "green"},
    "error": {"color": "red"},
    "info": {"color": "cyan"},
    "notice": {"color": "magenta"},
    "spam": {"color": "green", "faint": True},
    "success": {"bold": True, "color": "green"},
    "verbose": {"color": "blue"},
    "warning": {"color": "yellow"},
}
