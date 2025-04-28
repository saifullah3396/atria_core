"""
Common Utilities Module

This module provides common utility functions used throughout the Atria application.
These utilities include functions for handling partial objects, validating classes,
pretty-printing dictionaries, inspecting function arguments, and retrieving the
base path of the Atria package.

Functions:
    - _unwrap_partial: Unwraps a `functools.partial` object to retrieve the original function.
    - _validate_partial_class: Validates that a partial object is a subclass of a target class.
    - _msg_with_separator: Formats a message with a separator for better readability.
    - _pretty_print: Converts a dictionary or object to a pretty-printed string.
    - _get_possible_args: Retrieves all possible arguments of a function.
    - _get_required_args: Retrieves the required arguments of a function.

Dependencies:
    - functools.partial: For handling partially initialized objects.
    - inspect: For inspecting function signatures.
    - rich.pretty: For pretty-printing objects.
    - atria_core.logger.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Type

from atria_core.logger.logger import get_logger

logger = get_logger(__name__)


def _unwrap_partial(partial_object: partial) -> Any:
    """
    Unwraps a `functools.partial` object to retrieve the original function.

    Args:
        partial_object (partial): The partial object to unwrap.

    Returns:
        Any: The original function wrapped by the partial object.

    Raises:
        AssertionError: If the provided object is not a `functools.partial`.
    """
    assert isinstance(partial_object, partial)
    object = partial_object.func
    while hasattr(object, "__wrapped__"):
        object = object.__wrapped__
    return object


def _validate_partial_class(object: Any, target_class: Type[Any], object_name: str):
    """
    Validates that a partial object is a subclass of a target class.

    Args:
        object (Any): The object to validate.
        target_class (Type[Any]): The target class to validate against.
        object_name (str): The name of the object for error messages.

    Raises:
        AssertionError: If the object is not a partial or not a subclass of the target class.
    """
    assert isinstance(
        object, partial
    ), f"{object_name} must be a partial object of class {target_class} for late initialization"
    unwrapped = _unwrap_partial(object)
    if not callable(unwrapped):
        assert issubclass(
            unwrapped,
            target_class,
        ), f"{object_name} partial class must be a subclass of {target_class} or a callable function"


def _msg_with_separator(msg: str, separator: str = "=") -> str:
    """
    Formats a message with a separator for better readability.

    Args:
        msg (str): The message to format.
        separator (str): The character to use as a separator. Defaults to "=".

    Returns:
        str: The formatted message with separators.
    """
    separator = separator * (len(msg) + 8)
    return f"{separator} {msg} {separator}"


def _pretty_print(x: Any) -> str:
    """
    Converts a dictionary or object to a pretty-printed string.

    Args:
        x (Any): The dictionary or object to pretty-print.

    Returns:
        str: The pretty-printed string.
    """
    from rich.pretty import pretty_repr

    return pretty_repr(x)


def _get_possible_args(func: Any) -> inspect.Signature:
    """
    Retrieves all possible arguments of a function.

    Args:
        func (Any): The function to inspect.

    Returns:
        inspect.Signature: The signature of the function's parameters.
    """
    return inspect.signature(func).parameters


def _get_required_args(func: Any) -> list[str]:
    """
    Retrieves the required arguments of a function.

    Args:
        func (Any): The function to inspect.

    Returns:
        list[str]: A list of required argument names.
    """
    sig = inspect.signature(func)
    required_args = [
        param.name
        for param in sig.parameters.values()
        if param.default == inspect.Parameter.empty
        and param.kind
        in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
    ]
    return required_args
