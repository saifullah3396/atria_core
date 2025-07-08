"""
YAML Resolvers Utilities Module

This module provides utility functions and resolvers for use with OmegaConf. It includes
resolvers for sanitizing strings, generating directory names from overrides, and creating
tuples from arguments. These resolvers are registered with OmegaConf for dynamic usage
in YAML configurations.

Functions:
    - _sanitize_string: Replaces unsafe or unwanted characters in a string to make it file-system safe.
    - _resolve_dir_name: Sanitizes and simplifies input names for directory naming.
    - _dir_name_from_overrides: Generates a directory name from a list of overrides filtered by specified keys.
    - _resolve_tuple: Returns a tuple from a list of arguments.

Dependencies:
    - re: For regular expression operations.
    - omegaconf: For working with OmegaConf configurations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import re

from omegaconf import OmegaConf


def _sanitize_string(input_str: str) -> str:
    """
    Replaces unsafe or unwanted characters in a string to make it file-system safe.

    Args:
        input_str (str): The input string to sanitize.

    Returns:
        str: The sanitized string.
    """
    return re.sub(r"[{}\[\]/,]", "_", input_str).replace("=", "-")


def _resolve_dir_name(input_str: str) -> str:
    """
    Sanitizes and simplifies input names for directory naming.

    Args:
        input_str (str): The input string to resolve.

    Returns:
        str: The sanitized and simplified directory name.
    """
    if "." in input_str:
        input_str = input_str.split(".")[-1]
    return _sanitize_string(input_str)


def _resolve_tuple(*args) -> tuple:
    """
    Returns a tuple from a list of arguments.

    Args:
        *args: The arguments to convert into a tuple.

    Returns:
        Tuple: A tuple containing the provided arguments.
    """
    return tuple(args)


if not OmegaConf.has_resolver("resolve_dir_name"):
    OmegaConf.register_new_resolver("resolve_dir_name", _resolve_dir_name)


if not OmegaConf.has_resolver("as_tuple"):
    OmegaConf.register_new_resolver("as_tuple", _resolve_tuple)
