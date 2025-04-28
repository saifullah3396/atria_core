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

from typing import Any, List, Union

import torch
from pydantic import field_validator, model_validator

from atria_core.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel, BatchedBaseDataModel

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

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedLabel":
        """
        Constructs a new instance of BatchedLabel.

        Args:
            **kwargs: Arbitrary keyword arguments used to initialize the BatchedLabel.

        Returns:
            BatchedLabel: A new instance of BatchedLabel configured with the provided arguments.
        """
        return BatchedLabel(**kwargs)


class BatchedLabel(BatchedBaseDataModel):
    """
    A class for representing a batch of labels with values and names.

    Attributes:
        value (Union[List[int], torch.Tensor]): A 2D tensor where each row represents the value
            of labels in the batch. The shape of the tensor is (batch_size, num_labels).
        name (List[str]): A list of strings representing the names of the labels in the batch.

    Field Validators:
        - validate_value(value: Any) -> torch.Tensor:
            Ensures that the "value" field is a valid 2D tensor.
    """

    value: Union[List[int], torch.Tensor]
    name: List[str]

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the batch of labels to ensure it is a 1D tensor.

        Args:
            value (Any): The value to validate.

        Returns:
            torch.Tensor: The validated tensor.

        Raises:
            AssertionError: If the value is not a 1D tensor.
        """
        if isinstance(value, torch.Tensor):
            assert value.ndim == 1, f"Expected a 1D tensor, got {value.ndim}D tensor"
        return value


class LabelSequence(BaseDataModel):
    """
    A class for representing a sequence of labels with values and names.

    Attributes:
        value (Union[List[int], torch.Tensor]): The value of the labels in the sequence.
        names (List[str]): The names of the labels in the sequence.
    """

    values: Union[List[int], torch.Tensor]
    names: List[str]

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedLabelSequence":
        """
        Constructs a new instance of BatchedSequenceLabels.

        Args:
            **kwargs: Arbitrary keyword arguments used to initialize the BatchedSequenceLabels.

        Returns:
            BatchedSequenceLabels: A new instance of BatchedSequenceLabels configured with the provided arguments.
        """
        return BatchedLabelSequence(**kwargs)

    @classmethod
    def from_list(cls, value: List[Label]) -> "LabelSequence":
        """
        Constructs a new instance of SequenceLabels from a list of labels.

        Args:
            value (List[Label]): A list of Label instances.

        Returns:
            SequenceLabels: A new instance of SequenceLabels constructed from the list of labels.

        Raises:
            TypeError: If the input is not a list of Label instances.
        """
        assert len(value) > 0, "The list of labels cannot be empty"
        if isinstance(value, list) and all(isinstance(x, Label) for x in value):
            names = [x.name for x in value]
            values = [x.value for x in value]
            return cls(values=values, names=names)
        raise TypeError(f"Expected a list of Label instances, got {type(value)}")

    def to_list(self) -> List[Label]:
        """
        Converts the LabelSequence instance to a list of Label instances.

        Returns:
            List[Label]: A list of Label instances.
        """
        return [Label(value=v, name=n) for v, n in zip(self.values, self.names)]

    @model_validator(mode="after")
    def validate_shapes(self) -> "LabelSequence":
        """
        Validates the shapes of the value and names fields.

        Returns:
            SequenceLabels: The validated SequenceLabels instance.

        Raises:
            AssertionError: If the shapes of the value and names fields do not match.
        """
        if isinstance(self.values, torch.Tensor):
            assert (
                self.values.ndim == 1
            ), f"Expected a 1D tensor, got {self.values.ndim}D tensor"
        assert len(self.values) == len(
            self.names
        ), f"Expected the number of labels ({len(self.values)}) to match the number of names ({len(self.names)})"
        return self

    def __len__(self) -> int:
        """
        Returns the length of the sequence of labels.

        Returns:
            int: The length of the sequence of labels.
        """
        return len(self.values)

    def __getitem__(self, index: int) -> Label:
        """
        Returns the label at the specified index.

        Args:
            index (int): The index of the label to retrieve.

        Returns:
            Label: The label at the specified index.
        """
        return Label(value=self.values[index], name=self.names[index])


class BatchedLabelSequence(BatchedBaseDataModel):
    """
    A class for representing a batch of sequences of labels with values and names.

    Attributes:
        value (Union[List[List[int]], torch.Tensor]): A 3D tensor where each row represents the value
            of labels in the batch. The shape of the tensor is (batch_size, sequence_length, num_labels).
        names (List[List[str]]): A list of lists of strings representing the names of the labels in the batch.

    Field Validators:
        - validate_value(value: Any) -> torch.Tensor:
            Ensures that the "value" field is a valid 3D tensor.
    """

    values: Union[List[List[int]], torch.Tensor]
    names: List[List[str]]

    @field_validator("values", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the batch of sequences of labels.

        Args:
            value (Any): The value to validate.

        Returns:
            torch.Tensor: The validated tensor.

        Raises:
            AssertionError: If the value is not a valid 3D tensor.
        """
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 2
            ), f"Expected a 2D tensor of shape (B, L), got {value.ndim}D tensor"
        return value
