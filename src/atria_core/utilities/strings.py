"""
String Utilities Module

This module provides utility functions for handling and formatting strings. It includes
functions for indenting strings and converting camel case to snake case.

Functions:
    - _indent_string: Indents a given string by a specified number of spaces.
    - _convert_to_snake_case: Converts a camel case string to snake case.

Dependencies:
    - re: For regular expression operations.
    - textwrap: For string indentation.
    - typing: For type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import re
import textwrap


def _indent_string(s: str, ind: int = 4) -> str:
    """
    Indents a given string by a specified number of spaces.

    Args:
        s (str): The input string.
        ind (int): The number of spaces to indent. Defaults to 4.

    Returns:
        str: The indented string.
    """
    return textwrap.indent(s, " " * ind)


def _convert_to_snake_case(s: str) -> str:
    """
    Converts a camel case string to snake case.

    Args:
        s (str): The camel case string.

    Returns:
        str: The snake case string (underscored and lowercase).
    """
    return re.sub(r"([A-Z])", r"_\1", s).lower().lstrip("_")
