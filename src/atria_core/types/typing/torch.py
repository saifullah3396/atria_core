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

from typing import Annotated

import pyarrow as pa
import torch

from atria_core.types.typing.common import TableSchemaMetadata

"""
A type annotation for file paths.

Supports both `str` and `Path` types, with validation to ensure the path exists and is a file.
"""

TorchIntField = Annotated[int | torch.Tensor, TableSchemaMetadata(pyarrow=pa.int64())]
"""
An integer field type annotation with PyArrow metadata.
"""

TorchBoolField = Annotated[bool | torch.Tensor, TableSchemaMetadata(pyarrow=pa.bool_())]
"""
An integer field type annotation with PyArrow metadata.
"""

TorchFloatField = Annotated[
    float | torch.Tensor, TableSchemaMetadata(pyarrow=pa.float64())
]
"""
A float field type annotation with PyArrow metadata.
"""

TorchListIntField = Annotated[
    list[int] | torch.Tensor, TableSchemaMetadata(pyarrow=pa.list_(pa.int64()))
]
"""
A list of integers field type annotation with PyArrow metadata.
"""

TorchListFloatField = Annotated[
    list[float] | torch.Tensor, TableSchemaMetadata(pyarrow=pa.list_(pa.float64()))
]
"""
A list of floats field type annotation with PyArrow metadata.
"""

TorchListBoolField = Annotated[
    list[bool] | torch.Tensor, TableSchemaMetadata(pyarrow=pa.list_(pa.bool_()))
]
"""A list of booleans field type annotation with PyArrow metadata.
"""

###
# Optional fields
###

TorchOptIntField = Annotated[
    int | torch.Tensor | None, TableSchemaMetadata(pyarrow=pa.int64())
]
"""
An integer field type annotation with PyArrow metadata.
"""

TorchOptFloatField = Annotated[
    float | torch.Tensor | None, TableSchemaMetadata(pyarrow=pa.float64())
]
"""
A float field type annotation with PyArrow metadata.
"""

TorchOptListIntField = Annotated[
    list[int] | torch.Tensor | None, TableSchemaMetadata(pyarrow=pa.list_(pa.int64()))
]
"""
A list of integers field type annotation with PyArrow metadata.
"""

TorchOptListFloatField = Annotated[
    list[float] | torch.Tensor | None,
    TableSchemaMetadata(pyarrow=pa.list_(pa.float64())),
]
"""
A list of floats field type annotation with PyArrow metadata.
"""
