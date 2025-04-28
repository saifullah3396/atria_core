"""
Bounding Box Module

This module defines the `BoundingBox`, `SequenceBoundingBoxes`, and `BatchedSequenceBoundingBoxes` classes, which represent bounding boxes in an image or document.
It includes functionality for validating bounding box data, computing properties such as area, width, height, and normalizing bounding box coordinates.
The module also defines the `BoundingBoxMode` enum for specifying the format of bounding box coordinates.

Classes:
    - BoundingBoxMode: Enum for bounding box coordinate formats.
    - BoundingBox: A class for representing and manipulating bounding boxes.
    - SequenceBoundingBoxes: A class for representing a sequence of bounding boxes.
- BatchedSequenceBoundingBoxes: A class for representing a batch of sequences of bounding boxes.

Dependencies:
    - enum: For defining the `BoundingBoxMode` enum.
        - torch: For tensor operations.
    - pydantic: For data validation and serialization.
    - atria_core.data_types.base.data_model: For the base data model class.
    - atria_core.data_types.typing.tensor: For tensor type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import enum
from typing import Any, List, Union

import torch
from pydantic import field_validator

from atria_core.types.base.data_model import BaseDataModel, BatchedBaseDataModel


class BoundingBoxMode(str, enum.Enum):
    """
    Enum for bounding box coordinate formats.

    Attributes:
        XYXY (str): Format (x1, y1, x2, y2).
        XYWH (str): Format (x1, y1, width, height).
    """

    XYXY = "xyxy"  # (x1, y1, x2, y2)
    XYWH = "xywh"  # (x1, y1, width, height)


class BoundingBox(BaseDataModel):
    """
    A class for representing and manipulating bounding boxes.

    This class provides functionality for validating bounding box data, computing properties
    such as area, and normalizing bounding box coordinates.

    Attributes:
        value (Union[List[float],torch.Tensor]): The bounding box coordinates.
        mode (BoundingBoxMode): The format of the bounding box coordinates. Defaults to XYXY.
    """

    value: Union[List[float], torch.Tensor]
    mode: BoundingBoxMode = BoundingBoxMode.XYXY

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the bounding box.

        Ensures that the bounding box has the correct shape and format.

        Args:
            value (Any): The bounding box data to validate.

        Returns:
            torch.Tensor: The validated bounding box data.

        Raises:
            AssertionError: If the bounding box data is not in the correct format.
        """
        if isinstance(value, list):
            assert (
                len(value) == 4
            ), "Expected a 1D list of shape (4,) for bounding boxes."
        elif isinstance(value, torch.Tensor):
            assert (
                value.ndim == 1
            ), "Expected a 1D tensor of shape (4,) for bounding boxes."
            assert value.shape[-1] == 4, "Bounding boxes must have dimension (4,)"
        return value

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the bounding box tensor.

        Returns:
            torch.Size: The shape of the bounding box tensor.
        """
        self._validate_is_tensor()
        return self.value.shape

    @property
    def area(self) -> float:
        """
        Computes the area of the bounding box.

        Returns:
            float: The area of the bounding box.
        """
        return self.width * self.height

    @property
    def x1(self) -> float:
        """
        Returns the x1 coordinate of the bounding box.

        Returns:
            float: The x1 coordinate.
        """

        self._validate_is_tensor()
        return self.value[..., 0]

    @x1.setter
    def x1(self, value: float):
        """
        Sets the x1 coordinate of the bounding box.

        Args:
            value (float): The new x1 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 0] = value

    @property
    def y1(self) -> float:
        """
        Returns the y1 coordinate of the bounding box.

        Returns:
            float: The y1 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 1]

    @y1.setter
    def y1(self, value: float):
        """
        Sets the y1 coordinate of the bounding box.

        Args:
            value (float): The new y1 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 1] = value

    @property
    def x2(self) -> float:
        """
        Returns the x2 coordinate of the bounding box.

        Returns:
            float: The x2 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 2]

    @x2.setter
    def x2(self, value: float):
        """
        Sets the x2 coordinate of the bounding box.

        Args:
            value (float): The new x2 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 2] = value

    @property
    def y2(self) -> float:
        """
        Returns the y2 coordinate of the bounding box.

        Returns:
            float: The y2 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 3]

    @y2.setter
    def y2(self, value: float):
        """
        Sets the y2 coordinate of the bounding box.

        Args:
            value (float): The new y2 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 3] = value

    @property
    def width(self) -> float:
        """
        Computes the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """
        Computes the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        return self.y2 - self.y1

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedBoundingBox":
        """
        Constructs a batched instance of the model with the provided keyword arguments.

        This class method forwards all keyword arguments to the BatchedBoundingBox constructor
        and returns the newly created instance.

        Args:
            **kwargs: Arbitrary keyword arguments used to initialize the BatchedBoundingBox.

        Returns:
            BatchedBoundingBox: A new instance of BatchedBoundingBox configured with the provided arguments.
        """
        return BatchedBoundingBox(**kwargs)

    def normalize(self, width: float, height: float) -> "BoundingBox":
        """
        Normalizes the bounding box coordinates to the range [0, 1].

        Args:
            width (float): The width of the image or document.
            height (float): The height of the image or document.

        Returns:
            BoundingBox: The normalized bounding box.

        Raises:
            AssertionError: If the bounding box coordinates are invalid.
        """
        if self._is_tensor:
            assert width > 0, "Width must be greater than 0."
            assert height > 0, "Height must be greater than 0."
            assert self.x1 <= width, "x1 must be less than or equal to width."
            assert self.y1 <= height, "y1 must be less than or equal to height."
            assert self.x2 <= width, "x2 must be less than or equal to width."
            assert self.y2 <= height, "y2 must be less than or equal to height."
            self.x1 /= width
            self.y1 /= height
            self.x2 /= width
            self.y2 /= height
            return self
        else:
            self.value[0] /= width
            self.value[1] /= height
            self.value[2] /= width
            self.value[3] /= height
            return self


class BatchedBoundingBox(BatchedBaseDataModel):
    """
    A class for representing a batch of bounding boxes.

    This class extends the `BoundingBox` class to handle batches of bounding boxes.

    Attributes:
        value (torch.Tensor): A 2D tensor where each row represents a bounding box.
            The shape of the tensor is (batch_size, 4).
    """

    value: Union[List[List[float]], torch.Tensor]
    mode: List[BoundingBoxMode]

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the batch of bounding boxes after processing.

        Args:
            value (Any): The bounding box data to validate.

        Returns:
            torch.Tensor: The validated bounding box data.

        Raises:
            AssertionError: If the bounding box data is not in the correct format.
        """
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 2
            ), "Expected a 2D tensor of shape (batch size, 4) for bounding boxes."
            assert value.shape[-1] == 4, "Bounding boxes must have dimension (4,)"
        return value


class BoundingBoxSequence(BaseDataModel):
    """
        A class for representing a sequence of bounding boxes.

        This class extends the `BoundingBox` class to handle sequences of bounding boxes.

        Attributes:
            value (torch.Tensor): A 2D tensor where each row represents a bounding box.
                The shape of the tensor is (sequence_length, 4).
    mode (BoundingBoxMode): The format of the bounding box coordinates. Defaults to XYXY.
    """

    value: Union[List[List[float]], torch.Tensor]
    mode: BoundingBoxMode = BoundingBoxMode.XYXY

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the bounding box tensor.

        Returns:
            torch.Size: The shape of the bounding box tensor.
        """
        self._validate_is_tensor()
        return self.value.shape

    @property
    def area(self) -> float:
        """
        Computes the area of the bounding box.

        Returns:
            float: The area of the bounding box.
        """
        return self.width * self.height

    @property
    def x1(self) -> float:
        """
        Returns the x1 coordinate of the bounding box.

        Returns:
            float: The x1 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 0]

    @x1.setter
    def x1(self, value: float):
        """
        Sets the x1 coordinate of the bounding box.

        Args:
            value (float): The new x1 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 0] = value

    @property
    def y1(self) -> float:
        """
        Returns the y1 coordinate of the bounding box.

        Returns:
            float: The y1 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 1]

    @y1.setter
    def y1(self, value: float):
        """
        Sets the y1 coordinate of the bounding box.

        Args:
            value (float): The new y1 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 1] = value

    @property
    def x2(self) -> float:
        """
        Returns the x2 coordinate of the bounding box.

        Returns:
            float: The x2 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 2]

    @x2.setter
    def x2(self, value: float):
        """
        Sets the x2 coordinate of the bounding box.

        Args:
            value (float): The new x2 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 2] = value

    @property
    def y2(self) -> float:
        """
        Returns the y2 coordinate of the bounding box.

        Returns:
            float: The y2 coordinate.
        """
        self._validate_is_tensor()
        return self.value[..., 3]

    @y2.setter
    def y2(self, value: float):
        """
        Sets the y2 coordinate of the bounding box.

        Args:
            value (float): The new y2 coordinate.
        """
        self._validate_is_tensor()
        self.value[..., 3] = value

    @property
    def width(self) -> float:
        """
        Computes the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """
        Computes the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        return self.y2 - self.y1

    @classmethod
    def from_list(cls, value: List[BoundingBox]) -> "BoundingBoxSequence":
        """
        Constructs a new instance of SequenceBoundingBoxes from a list of bounding boxes.

        Args:
            value (List[BoundingBox]): A list of BoundingBox instances.

        Returns:
            SequenceBoundingBoxes: A new instance of SequenceBoundingBoxes constructed from the list of bounding boxes.

        Raises:
            TypeError: If the input is not a list of BoundingBox instances.
        """
        assert len(value) > 0, "The list of bounding boxes cannot be empty."
        if isinstance(value, list) and all(isinstance(x, BoundingBox) for x in value):
            assert all(
                item.mode == value[0].mode for item in value
            ), "All bounding boxes must have the same mode."
            return cls(value=[x.value for x in value], mode=value[0].mode)
        raise TypeError(f"Expected a list of BoundingBox instances, got {type(value)}")

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedBoundingBoxSequence":
        """
        Constructs a batched instance of the model with the provided keyword arguments.

        This class method forwards all keyword arguments to the BatchedSequenceBoundingBoxes constructor
        and returns the newly created instance.

        Args:
            **kwargs: Arbitrary keyword arguments used to initialize the BatchedSequenceBoundingBoxes.

        Returns:
            BatchedSequenceBoundingBoxes: A new instance of BatchedSequenceBoundingBoxes configured with the provided arguments.
        """
        return BatchedBoundingBoxSequence(**kwargs)

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the sequence of bounding boxes after processing.

        Args:
            value (Any): The bounding box data to validate.

        Returns:
            torch.Tensor: The validated bounding box data.

        Raises:
            AssertionError: If the bounding box data is not in the correct format.
        """
        if isinstance(value, list):
            assert (
                len(value) > 0
            ), "Expected a list of bounding boxes with at least one element."
            assert (
                len(value[0]) == 4
            ), "Expected a list of bounding boxes with shape (4,) for each bounding box."
        elif isinstance(value, torch.Tensor):
            assert (
                value.ndim == 2
            ), "Expected a 2D tensor of shape (sequence length, 4) for bounding boxes."
            assert value.shape[-1] == 4, "Bounding boxes must have dimension (4,)"
        return value

    def __len__(self) -> int:
        """
        Returns the length of the sequence of bounding boxes.

        Returns:
            int: The length of the sequence.
        """
        return len(self.value)

    def __getitem__(self, index: int) -> BoundingBox:
        """
        Returns the bounding box at the specified index.

        Args:
            index (int): The index of the bounding box to retrieve.

        Returns:
            BoundingBox: The bounding box at the specified index.
        """
        return BoundingBox(value=self.value[index], mode=self.mode)


class BatchedBoundingBoxSequence(BatchedBaseDataModel):
    """
        A class for representing a batch of sequences of bounding boxes.

        This class extends the `SequenceBoundingBoxes` class to handle batches of sequences
        of bounding boxes.

        Attributes:
            value (torch.Tensor): A 3D tensor where each row represents a sequence of bounding boxes.
                The shape of the tensor is (batch_size, sequence_length, 4).
    mode (List[BoundingBoxMode]): The format of the bounding box coordinates for each batch.
    """

    value: Union[List[List[List[float]]], torch.Tensor]
    mode: List[BoundingBoxMode]

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> torch.Tensor:
        """
        Validates the value of the batch of sequences of bounding boxes after processing.

        Args:
            value (Any): The bounding box data to validate.

        Returns:
            torch.Tensor: The validated bounding box data.

        Raises:
            AssertionError: If the bounding box data is not in the correct format.
        """
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 3
            ), "Expected a 3D tensor of shape (batch size, sequence length, 4) for bounding boxes."
            assert value.shape[-1] == 4, "Bounding boxes must have dimension (4,)"
        return value
