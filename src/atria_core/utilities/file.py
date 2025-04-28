"""
File Utilities Module

This module provides utility functions for handling file paths and resolving
them. It includes functionality for constructing full paths and optionally
validating their existence.

Functions:
    - _resolve_path: Constructs a full path by concatenating path segments and optionally validates its existence.

Dependencies:
    - pathlib.Path: For handling and resolving file paths.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from pathlib import Path


def _resolve_path(*args: str, validate: bool = True) -> Path:
    """
    Constructs a full path by concatenating all arguments and optionally validates its existence.

    Args:
        *args (str): Path segments to concatenate.
        validate (bool): If True, checks whether the constructed path exists.

    Returns:
        Path: The full constructed path.

    Raises:
        FileNotFoundError: If `validate=True` and the path does not exist.
    """
    full_path = Path(*args).resolve()

    if validate and not full_path.exists():
        raise FileNotFoundError(f"Path does not exist: {full_path}")

    return full_path
