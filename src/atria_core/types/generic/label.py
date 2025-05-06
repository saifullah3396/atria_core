"""
Label Module

This module defines classes for representing labels and their batched or sequential counterparts.
It includes functionality for validating label data, handling sequences of labels, and managing
batches of labels or sequences.

Classes:
    - Label: Represents a single label with a value and name.
    - BatchedLabel: Represents a batch of labels with values and names.
    - SequenceLabels: Represents a sequence of labels with values and names.
    - BatchedSequenceLabels: Represents a batch of sequences of labels with values and names.

Dependencies:
    - torch: For tensor operations.
    - pydantic: For data validation and serialization.
    - atria_core.logger: For logging utilities.
    - atria_core.utilities.tensors: For tensor stacking utilities.
    - atria_core.data_types.base.data_model: For the base data model class.
    - atria_core.data_types.typing.tensor: For tensor type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Any, Union

import torch
from pydantic import field_validator

from atria_core.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel

logger = get_logger(__name__)


class Label(BaseDataModel):
    """
    A class for representing a single label with a value and name.

    Attributes:
        value (Union[int, torch.Tensor]): The value of the label.
        name (str): The name of the label.
    """

    value: Union[int, torch.Tensor]
    name: str

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the label to ensure it is a scalar tensor.

        Args:
            value (Any): The value to validate.

        Returns:
            torch.Tensor: The validated tensor.

        Raises:
            AssertionError: If the value is not a scalar tensor.
        """
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 0
            ), f"Expected a scalar tensor, got {value.ndim}D tensor"
        return value
