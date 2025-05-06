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

from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import Field, field_serializer, field_validator

from atria_core.types.base.data_model import BaseDataModel

T = TypeVar("T", bound="BaseDataModel")


class BaseDataInstance(BaseDataModel):
    """
    A base class for individual data instances with UUIDs and utility methods.

    This class extends the `BaseDataModel` class to provide additional functionality
    for managing individual data instances. It includes support for unique identifiers (UUIDs),
    serialization, validation, and utility methods for pretty-printing and visualization.

    Attributes:
        id (UUID): A unique identifier for the data instance. Defaults to a randomly generated UUID.
    """

    id: UUID = Field(default_factory=uuid4)

    @field_serializer("id")
    def serialize_id(self, id: UUID, _info) -> str:
        """
        Serializes the UUID field to a string.

        Args:
            id (UUID): The UUID to serialize.
            _info: Additional information (unused).

        Returns:
            str: The serialized UUID as a string.
        """
        return str(id)

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value) -> UUID:
        """
        Validates and converts the UUID field.

        Args:
            value: The value to validate and convert.

        Returns:
            UUID: The validated and converted UUID.

        Raises:
            ValueError: If the value cannot be converted to a UUID.
        """
        if isinstance(value, list):
            return [UUID(str(v)) for v in value]
        return UUID(str(value))

    @property
    def key(self) -> str:
        """
        Generates a unique key for the data instance.

        The key is a combination of the UUID and the index (if present).

        Returns:
            str: The unique key for the data instance.
        """
        return str(self.id)

    def visualize(self) -> None:
        """
        Visualizes the data instance.

        This method is a placeholder and should be implemented by subclasses
        to provide specific visualization logic.
        """
