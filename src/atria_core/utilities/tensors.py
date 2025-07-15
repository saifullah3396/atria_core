"""
Tensor Utilities Module

This module provides utility functions for handling and converting data into PyTorch tensors.
It includes functionality for stacking tensors, converting various data types to tensors,
and checking if a nested list contains tensors.

Functions:
    - _stack_tensors_if_possible: Attempts to stack a list of tensors into a single tensor.
    - _convert_to_tensor: Converts various data types (e.g., lists, numbers, ndarrays) into PyTorch tensors.
    - _is_nested_list_of_tensors: Checks if a nested list contains tensors.

Dependencies:
    - numpy: For handling numerical arrays.
    - torch: For tensor operations.
    - atria_core.logger.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Optional, Union, cast

from pydantic import AfterValidator

from atria_core.logger import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def _stack_tensors_if_possible(
    tensors: list["torch.Tensor"],
) -> Union["torch.Tensor", list["torch.Tensor"]]:
    """
    Attempts to stack a list of tensors into a single tensor. If stacking is not possible,
    returns the list of tensors as-is.

    Args:
        tensors (list["torch.Tensor"]): A list of PyTorch tensors to stack.

    Returns:
        torch.Tensor | list[torch.Tensor]: A stacked tensor if possible, otherwise the original list of tensors.
    """
    import torch

    try:
        return torch.stack(tensors)
    except RuntimeError:
        return tensors


def _convert_to_tensor(value: Any) -> Union["torch.Tensor", list, str]:
    """
    Converts various data types (e.g., lists, numbers, ndarrays) into PyTorch tensors.

    Args:
        value (Any): The input value to convert. Can be a list, number, ndarray, or tensor.

    Returns:
        torch.Tensor | list | str: The converted PyTorch tensor, or the original value if conversion is not possible.

    Raises:
        TypeError: If the input is an empty list.
    """
    import numpy as np
    import torch
    from PIL.Image import Image as PILImage

    try:
        if isinstance(value, list):
            if len(value) == 0:
                return torch.tensor(value)
            if isinstance(value[0], list):
                value = [_convert_to_tensor(item) for item in value]
            if isinstance(value[0], numbers.Number):
                return torch.tensor(value)
            elif isinstance(value[0], torch.Tensor):
                return _stack_tensors_if_possible(value)
            elif isinstance(value[0], np.ndarray):
                return torch.from_numpy(np.array(value))
            elif isinstance(value[0], str):
                return value
        elif isinstance(value, PILImage):
            from torchvision.transforms.functional import to_tensor

            return to_tensor(value)
        elif isinstance(value, numbers.Number):
            return torch.tensor(value)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value
    except Exception as e:
        logger.warning(
            f"Failed to convert value {value} of type {type(value)} to tensor: {e}"
        )
        return value


def _convert_from_tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    elif isinstance(value, list):
        if len(value) == 0:
            return value
        if isinstance(value[0], torch.Tensor):
            return [_convert_from_tensor(v) for v in value]
    return value


def _validate_tensor_list_and_dtype(
    value: Any, dtype: Optional["torch.dtype"] = None
) -> None:
    import torch

    # if this is a list of tensors
    # this is possible when making a batch/list of non-uniform tensors
    assert all(isinstance(v, torch.Tensor) for v in value), (
        f"Expected a list of tensors, but got {type(value[0])}."
    )
    if dtype is not None:
        assert all(v.dtype == dtype for v in value), (
            f"Expected a list of {dtype} tensors, but got {[v.dtype for v in value]}."
        )


def _is_tensor_or_list_of_tensors(value: Any) -> bool:
    """
    Checks if a nested list contains tensors.

    Args:
        value (Any): The input value to check.

    Returns:
        bool: True if the nested list contains tensors, False otherwise.

    Raises:
        TypeError: If the input is an empty list.
    """
    if isinstance(value, list):
        return len(value) > 0 and isinstance(value[0], torch.Tensor)
    return isinstance(value, torch.Tensor)


def _apply_to_type(
    x: Any | Sequence | Mapping | str | bytes,
    input_type: type | tuple[type[Any], Any],
    func: Callable,
    strict: bool = True,
) -> Any | Sequence | Mapping | str | bytes:
    """Apply a function on an object of `input_type` or mapping, or sequence of objects of `input_type`.

    Args:
        x: object or mapping or sequence.
        input_type: data type of ``x``.
        func: the function to apply on ``x``.
    """
    if x is None or isinstance(x, input_type):
        return func(x)
    if isinstance(x, str | bytes):
        return x
    if isinstance(x, Mapping):
        return cast(Callable, type(x))(
            {k: _apply_to_type(sample, input_type, func) for k, sample in x.items()}
        )
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(
            *(_apply_to_type(sample, input_type, func) for sample in x)
        )
    if isinstance(x, Sequence):
        return cast(Callable, type(x))(
            [_apply_to_type(sample, input_type, func) for sample in x]
        )
    if strict:
        raise TypeError(f"x must contain {input_type}, dicts or lists; found {type(x)}")
    return x


def _convert_to_device(
    x: Any | Sequence | Mapping | str | bytes, device: "torch.device | str" = "cpu"
) -> Any | Sequence | Mapping | str | bytes:
    """Convert a tensor or a sequence of tensors to the specified device.

    Args:
        x: object or mapping or sequence.
        device: target device.

    Returns:
        The converted object, mapping, or sequence.
    """
    import torch

    return _apply_to_type(x, torch.Tensor, lambda t: t.to(device), strict=False)


def _validate_tensor(self):
    import torch

    if not isinstance(self, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(self).__name__}")
    return self


LazyTorchTensor = Annotated[Any, AfterValidator(_validate_tensor)]
