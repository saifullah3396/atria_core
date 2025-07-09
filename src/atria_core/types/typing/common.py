"""
Common Typing Module

This module defines common type annotations and utility functions for handling file paths,
tensor sizes, and tensor types. It includes custom serializers and validators for ensuring
the correctness of file paths and tensor sizes.

Type Annotations:
    - PydanticAtriaFilePath: A type annotation for file paths, supporting both `str` and `Path` types,
      with validation to ensure the path exists and is a file.
    - PydanticSize: A type annotation for tensor sizes, validated as `torch.Size`.

Functions:
    - _path_serializer: Serializes a file path to a string.
    - _size_validator: Validates and converts a value to `torch.Size`.
    - _path_validator: Validates a file path, ensuring it exists and is a file.

Dependencies:
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - torch: For tensor operations.
    - dacite: For handling generic types.
    - pydantic: For custom serializers and validators.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pyarrow as pa
from pydantic import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)


@dataclass
class TableSchemaMetadata:
    pyarrow: "pa.DataType"


def _path_serializer(value: str, nxt: SerializerFunctionWrapHandler) -> str:
    """
    Serializes a file path to a string.

    Args:
        value (str): The file path to serialize.
        nxt (SerializerFunctionWrapHandler): The next serializer in the chain.

    Returns:
        str: The serialized file path as a string.

    Raises:
        TypeError: If the value is not a valid file path.
    """
    if value is None:
        return None
    elif isinstance(value, Path):
        return nxt(str(value))
    elif isinstance(value, str):
        return nxt(value)


def _path_validator(
    value: str, handler: ValidatorFunctionWrapHandler
) -> Path | str | None:
    """
    Validates a file path, ensuring it exists and is a file.

    Args:
        value (str): The file path to validate.
        handler (ValidatorFunctionWrapHandler): The validation handler.

    Returns:
        Path: The validated file path.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the path is not a file.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, str):
        if value.startswith(("s3://", "http://", "https://")):
            return value
        return Path(value)
    return value


PydanticFilePath = Annotated[
    str | Path | None,
    WrapSerializer(_path_serializer),
    WrapValidator(_path_validator),
    TableSchemaMetadata(pyarrow=pa.string()),
]
"""
A type annotation for file paths.

Supports both `str` and `Path` types, with validation to ensure the path exists and is a file.
"""

IntField = Annotated[int, TableSchemaMetadata(pyarrow=pa.int64())]
"""
An integer field type annotation with PyArrow metadata.
"""

BoolField = Annotated[bool, TableSchemaMetadata(pyarrow=pa.bool_())]
"""
An integer field type annotation with PyArrow metadata.
"""

FloatField = Annotated[float, TableSchemaMetadata(pyarrow=pa.float64())]
"""
A float field type annotation with PyArrow metadata.
"""

ListIntField = Annotated[list[int], TableSchemaMetadata(pyarrow=pa.list_(pa.int64()))]
"""
A list of integers field type annotation with PyArrow metadata.
"""

ListFloatField = Annotated[
    list[float], TableSchemaMetadata(pyarrow=pa.list_(pa.float64()))
]
"""
A list of floats field type annotation with PyArrow metadata.
"""

StrField = Annotated[str, TableSchemaMetadata(pyarrow=pa.string())]
"""A list of strings field type annotation with PyArrow metadata.
"""

ListStrField = Annotated[list[str], TableSchemaMetadata(pyarrow=pa.list_(pa.string()))]
"""A list of strings field type annotation with PyArrow metadata.
"""
ListBoolField = Annotated[list[bool], TableSchemaMetadata(pyarrow=pa.list_(pa.bool_()))]
"""A list of booleans field type annotation with PyArrow metadata.
"""

###
# Optional fields
###

OptIntField = Annotated[int | None, TableSchemaMetadata(pyarrow=pa.int64())]
"""
An integer field type annotation with PyArrow metadata.
"""

OptFloatField = Annotated[float | None, TableSchemaMetadata(pyarrow=pa.float64())]
"""
A float field type annotation with PyArrow metadata.
"""

OptListIntField = Annotated[
    list[int] | None, TableSchemaMetadata(pyarrow=pa.list_(pa.int64()))
]
"""
A list of integers field type annotation with PyArrow metadata.
"""

OptListFloatField = Annotated[
    list[float] | None, TableSchemaMetadata(pyarrow=pa.list_(pa.float64()))
]
"""
A list of floats field type annotation with PyArrow metadata.
"""

OptStrField = Annotated[str | None, TableSchemaMetadata(pyarrow=pa.string())]
"""A list of strings field type annotation with PyArrow metadata.
"""

OptListStrField = Annotated[
    list[str] | None, TableSchemaMetadata(pyarrow=pa.list_(pa.string()))
]
"""A list of strings field type annotation with PyArrow metadata.
"""
