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
from pydantic import field_serializer, field_validator, model_validator

from atria_core.types.base.data_model import BaseDataModel, BatchedBaseDataModel
from atria_core.types.generic.bounding_box import (
    BatchedBoundingBox,
    BatchedBoundingBoxSequence,
    BoundingBox,
    BoundingBoxSequence,
)
from atria_core.types.generic.label import (
    BatchedLabel,
    BatchedLabelSequence,
    Label,
    LabelSequence,
)


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

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedAnnotatedObject":
        return BatchedAnnotatedObject(**kwargs)

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


class BatchedAnnotatedObject(BatchedBaseDataModel):
    """
    A class for representing a batch of annotated objects.

    This class extends the `AnnotatedObject` class to handle batched data. It supports
    multiple annotated objects represented by their labels, bounding boxes, segmentation masks,
    and crowd status.

    Attributes:
        label (Label): The label associated with the annotated object.
        bbox (BoundingBox): The bounding box of the object.
        segmentation (AtriaTensor): The segmentation mask of the object.
        iscrowd (bool): Indicates whether the object is part of a crowd. Defaults to False.
    """

    label: BatchedLabel
    bbox: BatchedBoundingBox
    segmentation: torch.Tensor
    iscrowd: List[bool]

    @field_validator("segmentation", mode="after")
    @classmethod
    def validate_segmentation(cls, value: Any) -> torch.Tensor:
        assert (
            value.ndim == 3
        ), "Segmentation must be of shape (B, N, 2M), where M is the number of points."
        return value


class AnnotatedObjectSequence(BaseDataModel):
    """
    A class for representing a sequence of annotated objects.

    This class extends the `BaseDataModel` class to include fields for a sequence of
    annotated objects. It supports multiple annotated objects represented by their labels,
    bounding boxes, segmentation masks, and crowd status.

    Attributes:
        label (Label): The label associated with the annotated object.
        bbox (BoundingBox): The bounding box of the object.
        segmentation (Union[List[List[List[float]]], torch.Tensor]): The segmentation mask of the object.
        iscrowd (bool): Indicates whether the object is part of a crowd. Defaults to False.
    """

    label: LabelSequence
    bbox: BoundingBoxSequence
    segmentation: Union[List[List[List[float]]], torch.Tensor] = None
    iscrowd: List[bool]

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: Any) -> LabelSequence:
        """
        Validates the label field.

        Args:
            value (Any): The value to validate.

        Returns:
            SequenceLabels: The validated SequenceLabels instance.

        Raises:
            AssertionError: If the value is not a valid SequenceLabels instance.
        """
        if isinstance(value, list):
            if all(isinstance(x, Label) for x in value):
                return LabelSequence.from_list(value)
            raise TypeError(f"Expected a list of Label instances, got {type(value)}")
        return value

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, value: Any) -> BoundingBoxSequence:
        """
        Validates the bounding box field.

        Args:
            value (Any): The value to validate.

        Returns:
            SequenceBoundingBoxes: The validated SequenceBoundingBoxes instance.

        Raises:
            AssertionError: If the value is not a valid SequenceBoundingBoxes instance.
        """
        if isinstance(value, list):
            if all(isinstance(x, BoundingBox) for x in value):
                return BoundingBoxSequence.from_list(value)
            raise TypeError(
                f"Expected a list of BoundingBox instances, got {type(value)}"
            )
        return value

    @field_validator("segmentation", mode="after")
    @classmethod
    def validate_segmentation(
        cls, value: Any
    ) -> Union[List[List[List[float]]], torch.Tensor]:
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 3
            ), "Segmentation must be of shape (L, N, 2M), where M is the number of points."
        return value

    @field_serializer("segmentation")
    @classmethod
    def serialize_segmentation(
        cls, segmentation: Union[List[List[List[float]]], torch.Tensor], _info
    ) -> List[List[List[float]]]:
        if isinstance(segmentation, torch.Tensor):
            return segmentation.tolist()
        return segmentation

    @model_validator(mode="after")
    def validate_shapes(self) -> "AnnotatedObjectSequence":
        """
        Validates the shapes of the labels, bounding boxes, and segmentation fields.

        Returns:
            SequenceAnnotatedObjects: The validated SequenceAnnotatedObjects instance.

        Raises:
            AssertionError: If the shapes of the value and names fields do not match.
        """
        assert (
            len(self.label)
            == len(self.bbox)
            == len(self.segmentation)
            == len(self.iscrowd)
        ), (
            f"Expected the number of labels ({len(self.label)}) to match the number of bounding boxes ({len(self.bbox)}) "
            f"and segmentation masks ({len(self.segmentation)}) and crowd status ({len(self.iscrowd)})"
        )
        return self

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedAnnotatedObjectSequence":
        """
        Constructs a new instance of BatchedSequenceAnnotatedObjects.

        Args:
            **kwargs: Keyword arguments for the constructor.

        Returns:
            BatchedSequenceAnnotatedObjects: A new instance of BatchedSequenceAnnotatedObjects.
        """
        return BatchedAnnotatedObjectSequence(**kwargs)

    @classmethod
    def from_list(cls, value: list[AnnotatedObject]) -> "AnnotatedObjectSequence":
        """
        Constructs a new instance of SequenceAnnotatedObjects from a list of labels.

        Args:
            value (List[AnnotatedObject]): A list of Label instances.

        Returns:
            SequenceAnnotatedObjects: A new instance of SequenceAnnotatedObjects constructed from the list of labels.

        Raises:
            TypeError: If the input is not a list of Label instances.
        """
        assert len(value) > 0, "The list of bounding boxes cannot be empty."
        if isinstance(value, list) and all(
            isinstance(x, AnnotatedObject) for x in value
        ):
            return cls(
                label=[x.label for x in value],
                bbox=[x.bbox for x in value],
                segmentation=[x.segmentation for x in value],
                iscrowd=[x.iscrowd for x in value],
            )
        raise TypeError(f"Expected a list of Label instances, got {type(value)}")

    def to_list(self) -> List[AnnotatedObject]:
        """
        Converts the SequenceAnnotatedObjects instance to a list of AnnotatedObject instances.

        Returns:
            List[AnnotatedObject]: A list of AnnotatedObject instances.
        """
        return [
            AnnotatedObject(
                label=self.label[i],
                bbox=self.bbox[i],
                segmentation=self.segmentation[i],
                iscrowd=self.iscrowd[i],
            )
            for i in range(len(self.label))
        ]

    def __len__(self) -> int:
        """
        Returns the length of the sequence of annotated objects.

        Returns:
            int: The length of the sequence of annotated objects.
        """
        return len(self.label)

    def __getitem__(self, index: int) -> AnnotatedObject:
        """
        Returns the annotated object at the specified index.

        Args:
            index (int): The index of the annotated object to retrieve.

        Returns:
            AnnotatedObject: The annotated object at the specified index.
        """
        return AnnotatedObject(
            label=self.label[index],
            bbox=self.bbox[index],
            segmentation=self.segmentation[index],
            iscrowd=self.iscrowd[index],
        )


class BatchedAnnotatedObjectSequence(BatchedBaseDataModel):
    """
    A class for representing a batch of sequences of annotated objects.

    This class extends the `SequenceAnnotatedObjects` class to handle batched data.
    It supports multiple sequences of annotated objects represented by their labels,
    bounding boxes, segmentation masks, and crowd status.

    Attributes:
        label (Label): The label associated with the annotated object.
        bbox (BoundingBox): The bounding box of the object.
        segmentation (AtriaTensor): The segmentation mask of the object.
        iscrowd (bool): Indicates whether the object is part of a crowd. Defaults to False.
    """

    label: BatchedLabelSequence
    bbox: BatchedBoundingBoxSequence
    segmentation: torch.Tensor
    iscrowd: List[List[bool]]

    @field_validator("segmentation", mode="after")
    @classmethod
    def validate_segmentation(cls, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 4
            ), "Segmentation must be of shape (B, L, N, 2M), where M is the number of points."
        return value
