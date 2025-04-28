"""
Image Typing Module

This module defines type annotations and utility functions for handling image tensors.
It includes custom serializers and validators for ensuring the correctness of image data,
supporting various formats such as base64-encoded strings, PIL images, NumPy arrays, and
PyTorch tensors.

Type Annotations:
    - PydanticImageTensor: A type annotation for image tensors, validated as `torch.Tensor`.

Functions:
    - _image_serializer: Serializes an image tensor to a base64-encoded string.
    - _image_validator: Validates and converts image data to a PyTorch tensor.

Dependencies:
    - typing: For type annotations.
    - numpy: For numerical operations.
    - PIL.Image: For handling image files.
    - torch: For tensor operations.
    - pydantic: For custom serializers and validators.
    - torchvision.transforms.functional: For converting images to tensors.
    - atria_core.utilities.encoding: For encoding and decoding image data.
    - atria_core.utilities.tensors: For tensor utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Annotated, Any

import numpy as np
import PIL
import PIL.Image
import torch
from pydantic import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)

from atria_core.utilities.encoding import _base64_to_image, _image_to_base64
from atria_core.utilities.tensors import (
    _is_tensor_or_list_of_tensors,
    _stack_tensors_if_possible,
)


def _image_serializer(value: Any, nxt: SerializerFunctionWrapHandler) -> str:
    """
    Serializes an image tensor to a base64-encoded string.

    Args:
        value (Any): The image tensor to serialize.
        nxt (SerializerFunctionWrapHandler): The next serializer in the chain.

    Returns:
        str: The serialized image as a base64-encoded string.
    """
    return nxt(_image_to_base64(value))


def _image_validator(value: Any, handler: ValidatorFunctionWrapHandler) -> torch.Tensor:
    """
    Validates and converts image data to a PyTorch tensor.

    Supports various formats such as base64-encoded strings, PIL images, NumPy arrays,
    and nested lists of tensors.

    Args:
        value (Any): The image data to validate and convert.
        handler (ValidatorFunctionWrapHandler): The validation handler.

    Returns:
        torch.Tensor: The validated and converted image tensor.

    Raises:
        AssertionError: If the image data is not in a valid format or has invalid dimensions.
    """
    from torchvision.transforms.functional import to_tensor

    if value is None:
        return value
    if isinstance(value, (bytes, str)):
        value = to_tensor(_base64_to_image(value))
    elif isinstance(value, (PIL.Image.Image, np.ndarray)):
        value = to_tensor(value)
    elif isinstance(value, list):
        value = _stack_tensors_if_possible(value)
        assert _is_tensor_or_list_of_tensors(
            value
        ), f"Expected a tensor or a lists of tensors, but got {type(value[0])}."
    if isinstance(value, torch.Tensor):
        assert value.ndim in [
            2,
            3,
            4,
        ], f"Invalid number of dimensions in the image tensor: {value.shape}. Image tensor must be 2D (grayscale) or 3D (channels, height, width)."
    return value


PydanticImageTensor = Annotated[
    torch.Tensor,
    WrapSerializer(_image_serializer),
    WrapValidator(_image_validator),
]
"""
A type annotation for image tensors.

Validated as `torch.Tensor` to ensure compatibility with tensor operations. Supports
various formats such as base64-encoded strings, PIL images, NumPy arrays, and nested
lists of tensors.
"""
