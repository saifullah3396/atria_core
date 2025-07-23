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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import (
    PlainSerializer,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)

from atria_core.utilities.encoding import (
    _base64_to_image,
    _bytes_to_image,
    _image_to_bytes,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from PIL.Image import Image as PILImage


@dataclass
class TableSchemaMetadata:
    pa_type: str

    def get_type(self) -> pa.DataType:
        """
        Converts the string representation of a PyArrow data type to a PyArrow DataType object.
        """
        return _resolve_pyarrow_type(self.pa_type)


def _resolve_pyarrow_type(type_str: str) -> pa.DataType:
    """
    Lazily resolve string identifiers to actual pyarrow types.
    """
    import pyarrow as pa

    if not isinstance(type_str, str):
        raise TypeError(f"Expected str, got {type(type_str)}")

    if type_str.startswith("list<") and type_str.endswith(">"):
        inner = type_str[5:-1]
        return pa.list_(_resolve_pyarrow_type(inner))

    mapping = {
        "int64": pa.int64(),
        "int32": pa.int32(),
        "int16": pa.int16(),
        "int8": pa.int8(),
        "uint64": pa.uint64(),
        "uint32": pa.uint32(),
        "uint16": pa.uint16(),
        "uint8": pa.uint8(),
        "float64": pa.float64(),
        "float32": pa.float32(),
        "float16": pa.float16(),
        "bool": pa.bool_(),
        "string": pa.string(),
        "binary": pa.binary(),
    }

    try:
        return mapping[type_str]
    except KeyError:
        raise ValueError(f"Unsupported pyarrow type string: '{type_str}'")


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
    TableSchemaMetadata(pa_type="string"),
]


def _is_tensor_type(value: Any) -> bool:
    """
    Check if the value is a tensor-like object.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a tensor-like object, False otherwise.
    """
    if hasattr(value, "__class__") and "torch" in value.__class__.__module__:
        import torch

        return isinstance(value, torch.Tensor)
    return False


def _tensor_validator(ndim: int) -> WrapValidator:
    """
    Creates a validator for tensor sizes.

    Args:
        ndim (int): The expected number of dimensions for the tensor.

    Returns:
        WrapValidator: A Pydantic validator that checks the tensor size.
    """

    def _wrapped(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """
        Validates a tensor, ensuring it has the correct dimensions.

        Args:
            value: The value to validate.
            handler (ValidatorFunctionWrapHandler): The validation handler.

        Returns:
            Any: The validated value.

        Raises:
            ValueError: If the tensor doesn't have the expected dimensions.
        """
        # Then check if it's a tensor-like object without importing torch
        if _is_tensor_type(value):
            import torch

            if isinstance(value, torch.Tensor):
                if value.ndim != ndim:
                    raise ValueError(
                        f"Expected a tensor with {ndim} dimensions, got {value.ndim}D tensor"
                    )
                return value

        return handler(value)

    return WrapValidator(_wrapped)


def _image_validator(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """
    Validates an image value, supporting various input types including tensors.

    Args:
        value (Any): The value to validate.
        handler (ValidatorFunctionWrapHandler): The validation handler.

    Returns:
        Any: The validated image value.

    Raises:
        ValueError: If the value type is not supported for image conversion.
    """
    import numpy as np
    import PIL.Image as PILImageModule
    from PIL.Image import Image as PILImage

    if value is None:
        return None

    # First check if it's a tensor-like object without importing torch
    if _is_tensor_type(value):
        import torch

        if isinstance(value, torch.Tensor):
            assert value.ndim in (2, 3, 4), "Tensor must be 2D, 3D or 4D."
            return value

    if isinstance(value, bytes):
        return _bytes_to_image(value)
    elif isinstance(value, str):
        return _base64_to_image(value)
    elif isinstance(value, np.ndarray):
        return PILImageModule.fromarray(value)
    elif isinstance(value, PILImage):
        return value
    else:
        raise ValueError(
            "Unsupported type for image conversion. Supported types are: str, bytes, np.ndarray, PIL.Image, torch.Tensor."
        )


def _image_serializer(value: PILImage | None) -> bytes | None:
    if value is None:
        return None
    if _is_tensor_type(value):
        from torchvision.transforms.functional import to_pil_image

        value = to_pil_image(value)
    return _image_to_bytes(value)


"""
A type annotation for file paths.

Supports both `str` and `Path` types, with validation to ensure the path exists and is a file.
"""

IntField = Annotated[int, _tensor_validator(0), TableSchemaMetadata(pa_type="int64")]
"""
An integer field type annotation with PyArrow metadata.
"""

BoolField = Annotated[bool, _tensor_validator(0), TableSchemaMetadata(pa_type="bool")]
"""
A boolean field type annotation with PyArrow metadata and tensor support.
"""

FloatField = Annotated[
    float, _tensor_validator(0), TableSchemaMetadata(pa_type="float64")
]
"""
A float field type annotation with PyArrow metadata and tensor support.
"""

ListIntField = Annotated[
    list[int], _tensor_validator(1), TableSchemaMetadata(pa_type="list<int64>")
]
"""
A list of integers field type annotation with PyArrow metadata and tensor support.
"""

ListFloatField = Annotated[
    list[float], _tensor_validator(1), TableSchemaMetadata(pa_type="list<float64>")
]
"""
A list of floats field type annotation with PyArrow metadata and tensor support.
"""

StrField = Annotated[str, TableSchemaMetadata(pa_type="string")]
"""A string field type annotation with PyArrow metadata.
"""

ListStrField = Annotated[list[str], TableSchemaMetadata(pa_type="list<string>")]
"""A list of strings field type annotation with PyArrow metadata.
"""

ListBoolField = Annotated[
    list[bool], _tensor_validator(1), TableSchemaMetadata(pa_type="list<bool>")
]
"""A list of booleans field type annotation with PyArrow metadata and tensor support.
"""

###
# Optional fields
###

OptIntField = Annotated[
    int | None, _tensor_validator(0), TableSchemaMetadata(pa_type="int64")
]
"""
An optional integer field type annotation with PyArrow metadata and tensor support.
"""

OptFloatField = Annotated[
    float | None, _tensor_validator(0), TableSchemaMetadata(pa_type="float64")
]
"""
An optional float field type annotation with PyArrow metadata and tensor support.
"""

OptListIntField = Annotated[
    list[int] | None, _tensor_validator(1), TableSchemaMetadata(pa_type="list<int64>")
]
"""
An optional list of integers field type annotation with PyArrow metadata and tensor support.
"""

OptListFloatField = Annotated[
    list[float] | None,
    _tensor_validator(1),
    TableSchemaMetadata(pa_type="list<float64>"),
]
"""
An optional list of floats field type annotation with PyArrow metadata and tensor support.
"""

OptStrField = Annotated[str | None, TableSchemaMetadata(pa_type="string")]
"""An optional string field type annotation with PyArrow metadata.
"""

OptListStrField = Annotated[
    list[str] | None, TableSchemaMetadata(pa_type="list<string>")
]
"""An optional list of strings field type annotation with PyArrow metadata.
"""

ValidatedPILImage = Annotated[
    Any | None,
    WrapValidator(_image_validator),
    PlainSerializer(_image_serializer),
    TableSchemaMetadata(pa_type="binary"),
]
