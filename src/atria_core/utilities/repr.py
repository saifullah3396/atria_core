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

from rich.pretty import RichReprResult, pretty_repr

from atria_core.constants import _MAX_REPR_PRINT_ELEMENTS


class RepresentationMixin:
    """
    Mixin class for rich representation of objects.

    This class provides methods for generating string representations of objects
    using the `rich` library. It includes support for both developer-friendly
    (`__repr__`) and human-readable (`__str__`) representations.
    """

    def __repr_name__(self) -> str:
        """
        Returns the name of the class for use in the `__repr__` method.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    def __rich_repr__(self) -> RichReprResult:
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        for name, field_repr in self.__dict__.items():
            if name is None:
                yield field_repr
            else:
                yield name, field_repr

    def __repr__(self) -> str:
        """
        Generates a developer-friendly string representation of the object.

        Returns:
            str: A developer-friendly string representation of the object.
        """
        return pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=4, max_depth=4
        )

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the object.

        Returns:
            str: A human-readable string representation of the object.
        """
        return pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=4, max_depth=4
        )
