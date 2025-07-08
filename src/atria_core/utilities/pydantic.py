# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: reportUnnecessaryTypeIgnoreComment=false

"""
Pydantic Utilities Module

This module provides utilities for integrating Pydantic validation with various
components of the Atria application. It includes functions for wrapping class
constructors, retrieving function signatures, and applying Pydantic parsing to
targets.

Functions:
    - _constructor_as_fn: Wraps a class constructor to make it compatible with Pydantic validation.
    - _get_signature: Retrieves the signature of a callable object.
    - pydantic_parser: Adds Pydantic parsing to a target for validation and type conversion.

Dependencies:
    - functools: For creating function wrappers.
    - inspect: For inspecting function signatures.
    - pydantic: For validation and type conversion.
    - atria_core.logger.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

import pydantic as _pyd

from atria_core.logger.logger import get_logger

_T = TypeVar("_T", bound=Callable[..., Any])


if _pyd.__version__ >= "2.0":  # pragma: no cover
    _default_parser = _pyd.validate_call(
        config={"arbitrary_types_allowed": True},
        validate_return=False,  # type: ignore
    )
else:  # pragma: no cover
    _default_parser = _pyd.validate_arguments(
        config={"arbitrary_types_allowed": True, "validate_return": False}  # type: ignore
    )

logger = get_logger(__name__)


def _constructor_as_fn(cls: Any) -> Any:
    """
    Wraps a class constructor to make it compatible with Pydantic validation.

    Notes:
        `pydantic.validate_call` mishandles class constructors; it expects that
        `cls`/`self` should be passed explicitly to the constructor. This shim
        corrects that.

    Args:
        cls (Any): The class to wrap.

    Returns:
        Any: A wrapped function compatible with Pydantic validation.
    """

    @functools.wraps(cls)
    def wrapper_function(*args, **kwargs):
        return cls(*args, **kwargs)

    annotations = getattr(cls, "__annotations__", {})

    # In a case like:
    # class A:
    #   x: int
    #   def __init__(self, y: int): ...
    #
    #  y will not be in __annotations__ but it should be in the signature,
    #  so we add it to the annotations.

    sig = inspect.signature(cls)
    for p, v in sig.parameters.items():
        if p not in annotations:
            annotations[p] = v.annotation
    wrapper_function.__annotations__ = annotations

    return wrapper_function


def _get_signature(x: Any) -> None | inspect.Signature:
    """
    Retrieves the signature of a callable object.

    Args:
        x (Any): The callable object.

    Returns:
        Union[None, inspect.Signature]: The signature of the callable object, or None if unavailable.
    """
    try:
        return inspect.signature(x)
    except Exception:
        return None


def pydantic_parser(target: _T, *, parser: Callable[[_T], _T] = _default_parser) -> _T:
    """
    Adds Pydantic parsing to a target for validation and type conversion.

    This can be passed to `instantiate` as a `_target_wrapper_` to add Pydantic parsing
    to the (recursive) instantiation of the target.

    Args:
        target (_T): The target callable to wrap.
        parser (Callable[[_T], _T], optional): A configured instance of Pydantic's validation decorator.
            Defaults to `_default_parser`.

    Returns:
        _T: The target callable with Pydantic parsing applied.

    Raises:
        RuntimeError: If Pydantic parsing fails to apply to the target.

    Examples:
        .. code-block:: python

           from hydra_zen import builds, instantiate
           from hydra_zen.third_party.pydantic import pydantic_parser

           from pydantic import PositiveInt


           def f(x: PositiveInt):
               return x


           good_conf = builds(f, x=10)
           bad_conf = builds(f, x=-3)

        >>> instantiate(good_conf, _target_wrapper_=pydantic_parser)
        10
        >>> instantiate(bad_conf, _target_wrapper_=pydantic_parser)
        ValidationError: 1 validation error for f (...)

        This also enables type conversion / parsing. E.g. Hydra can
        only produce lists from the CLI, but this parsing layer can
        convert them based on the annotated type. (Note: this only
        works for pydantic v2 and higher.)

        >>> def g(x: tuple):
        ...     return x
        >>> conf = builds(g, x=[1, 2, 3])
        >>> instantiate(conf, _target_wrapper_=pydantic_parser)
        (1, 2, 3)
    """
    try:
        if inspect.isbuiltin(target):
            return cast(_T, target)

        if isinstance(target, type) and issubclass(target, _pyd.BaseModel):
            # this already applies pydantic parsing
            return cast(_T, target)

        if not (_get_signature(target)):
            return cast(_T, target)

        if inspect.isclass(target):
            return cast(_T, parser(_constructor_as_fn(target)))

        return parser(target)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to apply pydantic parsing to {target}: {e}") from e
