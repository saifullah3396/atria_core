"""
Annotated Object Module

This module defines the `AnnotatedObject` class, which represents an object in an image
or document with associated metadata such as bounding boxes, segmentation masks, and labels.
It includes validation logic for segmentation data to ensure compatibility with expected formats.

Classes:
    - AnnotatedObject: A class for representing annotated objects with bounding boxes, segmentation, and labels.

Dependencies:
    - torch: For tensor operations.
    - pydantic: For data validation and serialization.
    - atria_core.data_types.base.data_model: For the base data model class.
    - atria_core.data_types.generic.bounding_box: For bounding box structures.
    - atria_core.data_types.generic.label: For label structures.
    - atria_core.data_types.typing.tensor: For tensor type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Any, List, Union

import torch
from pydantic import field_serializer, field_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.label import Label


class AnnotatedObject(BaseDataModel):
    """
    A class for representing annotated objects with bounding boxes, segmentation, and labels.

    This class extends the `BaseDataModel` class to include fields for bounding boxes,
    segmentation masks, and labels. It also includes validation logic for segmentation data
    to ensure compatibility with expected formats.

    Attributes:
        label (Label): The label associated with the annotated object.
        bbox (BoundingBox): The bounding box of the object.
        segmentation (torch.Tensor]): The segmentation mask of the object.
        iscrowd (bool): Indicates whether the object is part of a crowd. Defaults to False.
    """

    label: Label
    bbox: BoundingBox
    segmentation: Union[List[List[float]], torch.Tensor] = None
    iscrowd: bool = False

    @field_validator("segmentation", mode="before")
    @classmethod
    def validate_and_convert_segmentation(cls, value: Any) -> torch.Tensor:
        """
        Validates and converts the segmentation field.

        The segmentation can be provided as a dictionary with "counts" and "size",
        a list of polygons, or a tensor. This method ensures that the segmentation
        data is converted to a tensor and meets the expected format.

        Args:
            v: The segmentation data to validate and convert.

        Returns:
            torch.Tensor: The validated and converted segmentation tensor.

        Raises:
            ValueError: If the segmentation data is not in a valid format.
        """
        if isinstance(value, dict) and "counts" in value and "size" in value:
            return value["counts"]
        elif isinstance(value, list) and all(isinstance(poly, list) for poly in value):
            tensor_seg = torch.tensor(value, dtype=torch.float32)
            if tensor_seg.dim() != 2 or tensor_seg.shape[1] % 2 != 0:
                raise ValueError(
                    "Polygon segmentation must be of shape (N, 2M), where M is the number of points."
                )
            return tensor_seg.tolist()
        elif isinstance(value, torch.Tensor):
            if value.dim() != 2 or value.shape[1] % 2 != 0:
                raise ValueError(
                    "Polygon segmentation must be of shape (N, 2M), where M is the number of points."
                )
            return value
        else:
            raise ValueError(
                "Segmentation must be a list of polygons, a dictionary with 'counts' and 'size', or a tensor."
            )

    @field_serializer("segmentation")
    @classmethod
    def serialize_segmentation(
        cls, segmentation: Union[List[List[float]], torch.Tensor], _info
    ) -> List[List[float]]:
        if isinstance(segmentation, torch.Tensor):
            return segmentation.tolist()
        return segmentation
