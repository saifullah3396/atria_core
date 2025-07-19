"""
Rich Representation Utilities Module

This module provides a mixin class for rich representation of objects. It uses
the `rich` library to generate human-readable and developer-friendly string
representations for objects.

Classes:
    - RepresentationMixin: A mixin class for providing rich representations of objects.

Dependencies:
    - rich.pretty.RichReprResult: For generating rich representations of objects.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import types

from rich.pretty import RichReprResult, pretty_repr

from atria_core.constants import _MAX_REPR_PRINT_ELEMENTS


class RepresentationMixin:
    """
    Mixin class for rich representation of objects.

    This class provides methods for generating string representations of objects
    using the `rich` library. It includes support for both developer-friendly
    (`__repr__`) and human-readable (`__str__`) representations.

    Set `__repr_fields__` as a class attribute to specify which fields to include.
    """

    __repr_fields__: set[str] = set()

    def __repr_name__(self) -> str:
        """
        Returns the name of the class for use in the `__repr__` method.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    def __rich_repr__(self) -> RichReprResult:  # type: ignore
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs for the specified fields only.
        """
        repr_fields = getattr(self.__class__, "__repr_fields__", set())

        for field_name in repr_fields:
            if not hasattr(self, field_name):
                continue

            value = getattr(self, field_name)

            # Safely represent bound methods, functions, or other callables
            if isinstance(value, types.MethodType):
                safe_value = value.__func__
            else:
                safe_value = value
            if safe_value is not None:
                yield field_name, safe_value

    def __repr__(self) -> str:
        """
        Generates a developer-friendly string representation of the object.

        Returns:
            str: A developer-friendly string representation of the object.
        """

        return pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=128, max_depth=4
        )

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the object.

        Returns:
            str: A human-readable string representation of the object.
        """
        return pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=128, max_depth=4
        )
