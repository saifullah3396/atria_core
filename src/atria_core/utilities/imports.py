"""
Imports Utilities Module

This module provides utility functions for dynamically importing modules and resolving
module paths. It includes functionality for retrieving the base path of the Atria package,
resolving classes or functions from module paths, and retrieving parent module names.

Functions:
    - _get_package_base_path: Retrieves the base path of a specified package.
    - _get_atria_base_path: Retrieves the base path of the Atria package.
    - _resolve_module_from_path: Resolves a class or function from a module path string.
    - _get_parent_module: Retrieves the parent module name from a given module name.

Dependencies:
    - importlib: For dynamically importing modules.
    - pathlib.Path: For handling file paths.
    - atria_core.logger.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import importlib
from importlib import import_module
from pathlib import Path

from atria_core.logger import get_logger

logger = get_logger(__name__)


def _get_package_base_path(package: str) -> str | None:
    """
    Retrieves the base path of the specified package.

    Args:
        package (str): The name of the package.

    Returns:
        str | None: The base path of the specified package as a string, or None if the package is not found.
    """
    spec = importlib.util.find_spec(package)
    return str(Path(spec.origin).parent) if spec else None


def _get_atria_base_path() -> str | None:
    """
    Retrieves the base path of the Atria package.

    Returns:
        str | None: The base path of the Atria package as a string, or None if the package is not found.
    """
    return _get_package_base_path("atria")


def _resolve_module_from_path(module_path: str) -> object:
    """
    Resolves a class or function from a module path string.

    Args:
        module_path (str): The module path in the format 'module_name.class_name'.

    Returns:
        object: The resolved class or function.

    Raises:
        ValueError: If the module path is invalid or cannot be resolved.
    """
    path = module_path.rsplit(".", 1)
    if len(path) == 1:
        raise ValueError(
            f"Invalid module path: {module_path}. It should be in the form 'module_name.class_name'."
        )
    module_name, class_name = path
    module = import_module(module_name)
    return getattr(module, class_name)


def _get_parent_module(module_name: str) -> str:
    """
    Retrieves the parent module name from a given module name.

    Args:
        module_name (str): The name of the module.

    Returns:
        str: The parent module name. If the module has no parent, returns the module name itself.
    """
    return module_name.rsplit(".", 1)[0] if "." in module_name else module_name
