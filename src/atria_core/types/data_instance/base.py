"""
Base Data Instance Module

This module defines the `BaseDataInstance` class, which extends the `BaseDataModel` class
to provide additional functionality for managing individual data instances. It includes
support for unique identifiers (UUIDs), serialization, validation, and utility methods
for pretty-printing and visualization.

Classes:
    - BaseDataInstance: A base class for individual data instances with UUIDs and utility methods.

Dependencies:
    - typing: For type annotations.
    - uuid: For generating and managing unique identifiers.
    - pydantic: For data validation and serialization.
    - rich.pretty: For pretty-printing representations.
    - atria_core.data_types.base.data_model: For the base data model class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import ClassVar

from atria_core.types.base.data_model import BaseDataModel, RowSerializable


class BaseDataInstance(BaseDataModel, RowSerializable):
    """
    A base class for individual data instances with UUIDs and utility methods.

    This class extends the `BaseDataModel` class to provide additional functionality
    for managing individual data instances. It includes support for unique identifiers (UUIDs),
    serialization, validation, and utility methods for pretty-printing and visualization.

    Attributes:
        id (UUID): A unique identifier for the data instance. Defaults to a randomly generated UUID.
    """

    _row_serialization_types: ClassVar[dict[str, str]] = {
        "index": int,
        "sample_id": str,
    }

    index: int | None = None
    sample_id: str

    @property
    def key(self) -> str:
        """
        Generates a unique key for the data instance.

        The key is a combination of the UUID and the index (if present).

        Returns:
            str: The unique key for the data instance.
        """
        return str(self.sample_id)

    def visualize(self) -> None:
        """
        Visualizes the data instance.

        This method is a placeholder and should be implemented by subclasses
        to provide specific visualization logic.
        """
