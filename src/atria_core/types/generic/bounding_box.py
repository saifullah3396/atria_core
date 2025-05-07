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
from atria_core.types.base.data_model import BaseDataModel, BaseDataModelConfigDict
from pydantic import field_validator


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

    model_config = BaseDataModelConfigDict(batch_merge_fields=["mode"])

    value: Union[List[float], torch.Tensor]
    mode: BoundingBoxMode = BoundingBoxMode.XYXY

    def switch_mode(self):
        assert not self._is_batched, "Cannot switch mode for batched bounding boxes."
        if self.mode == BoundingBoxMode.XYXY:
            self.value = [self.x1, self.y1, self.width, self.height]
            self.mode = BoundingBoxMode.XYWH
        else:
            self.value = [self.x1, self.y1, self.x2, self.y2]
            self.mode = BoundingBoxMode.XYXY
        if self._is_tensor:
            self.value = torch.tensor(self.value)
        return self

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
    def is_valid(self) -> bool:
        """
        Checks if the bounding box is valid.

        Returns:
            bool: True if the bounding box is valid, False otherwise.
        """
        return (
            self.x1 >= 0
            and self.y1 >= 0
            and self.x2 > self.x1
            and self.y2 > self.y1
            and self.width > 0
            and self.height > 0
        )

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the bounding box tensor.

        Returns:
            torch.Size: The shape of the bounding box tensor.
        """
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
        idx = (..., 0) if self._is_tensor else 0
        return self.value[idx]

    @x1.setter
    def x1(self, value: float):
        """
        Sets the x1 coordinate of the bounding box.

        Args:
            value (float): The new x1 coordinate.
        """
        idx = (..., 0) if self._is_tensor else 0
        self.value[idx] = value

    @property
    def y1(self) -> float:
        """
        Returns the y1 coordinate of the bounding box.

        Returns:
            float: The y1 coordinate.
        """
        idx = (..., 1) if self._is_tensor else 1
        return self.value[idx]

    @y1.setter
    def y1(self, value: float):
        """
        Sets the y1 coordinate of the bounding box.

        Args:
            value (float): The new y1 coordinate.
        """
        idx = (..., 1) if self._is_tensor else 1
        self.value[idx] = value

    @property
    def x2(self) -> float:
        """
        Returns the x2 coordinate of the bounding box.

        Returns:
            float: The x2 coordinate.
        """
        if self.mode == BoundingBoxMode.XYWH:
            return self.x1 + self.width
        else:
            idx = (..., 2) if self._is_tensor else 2
            return self.value[idx]

    @x2.setter
    def x2(self, value: float):
        """
        Sets the x2 coordinate of the bounding box.

        Args:
            value (float): The new x2 coordinate.
        """
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            idx = (..., 2) if self._is_tensor else 2
            self.value[idx] = value

    @property
    def y2(self) -> float:
        """
        Returns the y2 coordinate of the bounding box.

        Returns:
            float: The y2 coordinate.
        """
        if self.mode == BoundingBoxMode.XYWH:
            return self.y1 + self.height
        else:
            idx = (..., 3) if self._is_tensor else 3
            return self.value[idx]

    @y2.setter
    def y2(self, value: float):
        """
        Sets the y2 coordinate of the bounding box.

        Args:
            value (float): The new y2 coordinate.
        """
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            idx = (..., 3) if self._is_tensor else 3
            self.value[idx] = value

    @property
    def width(self) -> float:
        """
        Computes the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 2) if self._is_tensor else 2
            return self.value[idx]
        else:
            return self.x2 - self.x1

    @width.setter
    def width(self, value: float):
        """
        Sets the width of the bounding box.

        Args:
            value (float): The new width of the bounding box.
        """
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 2) if self._is_tensor else 2
            self.value[idx] = value
        else:
            raise ValueError("Cannot set width directly in XYXY mode. Use x2 instead.")

    @property
    def height(self) -> float:
        """
        Computes the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 3) if self._is_tensor else 3
            return self.value[idx]
        else:
            return self.y2 - self.y1

    @height.setter
    def height(self, value: float):
        """
        Sets the height of the bounding box.

        Args:
            value (float): The new height of the bounding box.
        """
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 3) if self._is_tensor else 3
            self.value[idx] = value
        else:
            raise ValueError("Cannot set height directly in XYXY mode. Use y2 instead.")

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
        assert width > 0, "Width must be greater than 0."
        assert height > 0, "Height must be greater than 0."
        assert self.x1 <= width, "x1 must be less than or equal to width."
        assert self.y1 <= height, "y1 must be less than or equal to height."
        assert self.x2 <= width, "x2 must be less than or equal to width."
        assert self.y2 <= height, "y2 must be less than or equal to height."
        if self.mode == BoundingBoxMode.XYWH:
            self.x1 /= width
            self.y1 /= height
            self.width /= width
            self.height /= height
        else:
            self.x1 /= width
            self.y1 /= height
            self.x2 /= width
            self.y2 /= height
        return self
