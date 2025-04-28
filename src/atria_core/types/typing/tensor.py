"""
Tensor Typing Module

This module defines type annotations and utility functions for handling PyTorch tensors.
It includes custom serializers and validators for ensuring the correctness of tensor data,
supporting various data types such as float, double, int, and boolean tensors.

Type Annotations:
    - PydanticDoubleTensor: A type annotation for double-precision tensors.
    - PydanticFloatTensor: A type annotation for single-precision float tensors.
    - PydanticBFloat16Tensor: A type annotation for bfloat16 tensors.
    - PydanticLongTensor: A type annotation for long integer tensors.
    - PydanticIntTensor: A type annotation for integer tensors.
    - PydanticShortTensor: A type annotation for short integer tensors.
    - PydanticHalfTensor: A type annotation for half-precision float tensors.
    - PydanticByteTensor: A type annotation for byte tensors.
    - PydanticBoolTensor: A type annotation for boolean tensors.
    - PydanticTensor: A generic type annotation for tensors of any data type.
    - OptionalPydantic*: Variants of the above types that allow `None`.

Functions:
    - _tensor_serializer: Serializes a tensor to a NumPy array or a list.
    - _tensor_validator: Validates and converts data to a PyTorch tensor.

Dependencies:
    - typing: For type annotations.
    - numpy: For numerical operations.
    - torch: For tensor operations.
    - pydantic: For custom serializers and validators.
    - atria_core.utilities.tensors: For tensor conversion and validation utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Annotated, Any

import numpy as np
import torch
from pydantic import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)

from atria_core.utilities.tensors import (
    _convert_to_tensor,
    _validate_tensor_list_and_dtype,
)


def _tensor_serializer(value: Any, nxt: SerializerFunctionWrapHandler) -> np.ndarray:
    """
    Serializes a tensor to a NumPy array or a list.

    Args:
        value (Any): The tensor to serialize.
        nxt (SerializerFunctionWrapHandler): The next serializer in the chain.

    Returns:
        np.ndarray: The serialized tensor as a NumPy array or list.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return nxt([v.detach().cpu().tolist() for v in value])
    return nxt(value.detach().cpu().tolist())


def _tensor_validator(
    dtype: torch.dtype, required: bool = True
) -> ValidatorFunctionWrapHandler:
    """
    Creates a validator function for tensors of a specific data type.

    Args:
        dtype (torch.dtype): The expected data type of the tensor.
        required (bool): Whether the tensor is required. Defaults to True.

    Returns:
        Callable: A validator function for the specified tensor data type.

    Raises:
        AssertionError: If the tensor data type or structure is invalid.
    """

    def wrapped(value: Any, handler: ValidatorFunctionWrapHandler) -> torch.Tensor:
        """
        Validates and converts data to a PyTorch tensor.

        Args:
            value (Any): The data to validate and convert.
            handler (ValidatorFunctionWrapHandler): The validation handler.

        Returns:
            torch.Tensor: The validated and converted tensor.

        Raises:
            AssertionError: If the data is not a valid tensor or does not match the expected data type.
        """
        if value is None:
            assert not required, f"Expected a tensor, but got None."
            return None
        value = _convert_to_tensor(value)
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                # if this is a list of lists, which it is a list of list of tensors
                # this is possible when batching a non-uniform list of tensors
                for v in value:
                    _validate_tensor_list_and_dtype(v, dtype)
            else:
                # if this is a list of tensors
                # this is possible when making a batch/list of non-uniform tensors
                _validate_tensor_list_and_dtype(value, dtype)
        else:
            assert isinstance(
                value, torch.Tensor
            ), f"Expected a tensor, but got {type(value)}."
            if dtype is not None:
                assert (
                    value.dtype == dtype
                ), f"Expected a {dtype} tensor, but got {value.dtype}."
        return value

    return wrapped


# Type annotations for specific tensor types
PydanticDoubleTensor = Annotated[
    torch.DoubleTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.double)),
]
"""
A type annotation for double-precision tensors.

Validated as `torch.DoubleTensor` to ensure compatibility with double-precision operations.
"""

PydanticFloatTensor = Annotated[
    torch.FloatTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.float)),
]
"""
A type annotation for single-precision float tensors.

Validated as `torch.FloatTensor` to ensure compatibility with single-precision operations.
"""

PydanticBFloat16Tensor = Annotated[
    torch.BFloat16Tensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.bfloat16)),
]
"""
A type annotation for bfloat16 tensors.

Validated as `torch.BFloat16Tensor` to ensure compatibility with bfloat16 operations.
"""

PydanticLongTensor = Annotated[
    torch.LongTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.long)),
]
"""
A type annotation for long integer tensors.

Validated as `torch.LongTensor` to ensure compatibility with long integer operations.
"""

PydanticIntTensor = Annotated[
    torch.IntTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.int)),
]
"""
A type annotation for integer tensors.

Validated as `torch.IntTensor` to ensure compatibility with integer operations.
"""

PydanticShortTensor = Annotated[
    torch.ShortTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.short)),
]
"""
A type annotation for short integer tensors.

Validated as `torch.ShortTensor` to ensure compatibility with short integer operations.
"""

PydanticHalfTensor = Annotated[
    torch.HalfTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.half)),
]
"""
A type annotation for half-precision float tensors.

Validated as `torch.HalfTensor` to ensure compatibility with half-precision operations.
"""

PydanticByteTensor = Annotated[
    torch.ByteTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.uint8)),
]
"""
A type annotation for byte tensors.

Validated as `torch.ByteTensor` to ensure compatibility with byte operations.
"""

PydanticBoolTensor = Annotated[
    torch.BoolTensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.bool)),
]
"""
A type annotation for boolean tensors.

Validated as `torch.BoolTensor` to ensure compatibility with boolean operations.
"""

PydanticTensor = Annotated[
    torch.Tensor,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=None)),
]
"""
A generic type annotation for tensors of any data type.

Validated as `torch.Tensor` to ensure compatibility with tensor operations.
"""

# Optional type annotations for tensors
OptionalPydanticDoubleTensor = Annotated[
    torch.DoubleTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.double, required=False)),
]
"""
A type annotation for double-precision tensors that can be None.

Validated as `torch.DoubleTensor` to ensure compatibility with double-precision operations.
"""

OptionalPydanticFloatTensor = Annotated[
    torch.FloatTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.float, required=False)),
]
"""
A type annotation for single-precision float tensors that can be None.

Validated as `torch.FloatTensor` to ensure compatibility with single-precision operations.
"""

OptionalPydanticBFloat16Tensor = Annotated[
    torch.BFloat16Tensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.bfloat16, required=False)),
]
"""
A type annotation for bfloat16 tensors that can be None.

Validated as `torch.BFloat16Tensor` to ensure compatibility with bfloat16 operations.
"""

OptionalPydanticLongTensor = Annotated[
    torch.LongTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.long, required=False)),
]
"""
A type annotation for long integer tensors that can be None.

Validated as `torch.LongTensor` to ensure compatibility with long integer operations.
"""

OptionalPydanticIntTensor = Annotated[
    torch.IntTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.int, required=False)),
]
"""
A type annotation for integer tensors that can be None.

Validated as `torch.IntTensor` to ensure compatibility with integer operations.
"""

OptionalPydanticShortTensor = Annotated[
    torch.ShortTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.short, required=False)),
]
"""
A type annotation for short integer tensors that can be None.

Validated as `torch.ShortTensor` to ensure compatibility with short integer operations.
"""

OptionalPydanticHalfTensor = Annotated[
    torch.HalfTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.half, required=False)),
]
"""
A type annotation for half-precision float tensors that can be None.

Validated as `torch.HalfTensor` to ensure compatibility with half-precision operations.
"""

OptionalPydanticByteTensor = Annotated[
    torch.ByteTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.uint8, required=False)),
]
"""
A type annotation for byte tensors that can be None.

Validated as `torch.ByteTensor` to ensure compatibility with byte operations.
"""

OptionalPydanticBoolTensor = Annotated[
    torch.BoolTensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=torch.bool, required=False)),
]
"""
A type annotation for boolean tensors that can be None.

Validated as `torch.BoolTensor` to ensure compatibility with boolean operations.
"""

OptionalPydanticTensor = Annotated[
    torch.Tensor | None,
    WrapSerializer(_tensor_serializer),
    WrapValidator(_tensor_validator(dtype=None, required=False)),
]
"""
A generic type annotation for tensors of any data type that can be None.

Validated as `torch.Tensor` to ensure compatibility with tensor operations.
"""
