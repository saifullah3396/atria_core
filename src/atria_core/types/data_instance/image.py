"""
Image Instance Module

This module defines the `ImageInstance` and `BatchedImageInstance` classes, which extend the `BaseDataInstance` class
to represent images with associated metadata and annotations. These classes include fields for the image data and
its corresponding label, supporting both single and batched instances.

Classes:
    - ImageInstance: A class for representing a single image with associated metadata and annotations.
    - BatchedImageInstance: A class for representing a batch of image instances.

Dependencies:
    - atria_core.data_types.data_instance.base: For the base data instance class.
    - atria_core.data_types.generic.image: For image structures.
    - atria_core.data_types.generic.label: For label structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import ClassVar

from atria_core.types.base.data_model import RowSerializable
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image


class ImageInstance(BaseDataInstance, RowSerializable):
    """
    A class for representing a single image with associated metadata and annotations.

    This class extends the `BaseDataInstance` class to include fields for the image data
    and its corresponding label.

    Attributes:
        image (Image): The image data associated with the instance.
        ground_truth (GroundTruth): The ground truth associated with the image.
    """

    _row_name: ClassVar[str | None] = None
    _row_serialization_types: ClassVar[dict[str, str]] = {
        **BaseDataInstance.row_serialization_types(),
        **Image.row_serialization_types(),
        **GroundTruth.row_serialization_types(),
    }

    image: Image
    ground_truth: GroundTruth = GroundTruth()

    def to_row(self) -> dict:
        """
        Converts the instance to a row format suitable for storage or serialization.
        Returns:
            dict: A dictionary representation of the instance, with keys prefixed by "image_", "ground_truth_", and "ocr_".
        """
        model_dict = self.model_dump(exclude={"ground_truth", "image"})
        row = {
            **model_dict,
            **self.image.to_row(),
            **self.ground_truth.to_row(),
        }
        return row

    @classmethod
    def from_row(cls, row: dict) -> "ImageInstance":
        """
        Creates an instance from a row format.

        Args:
            row (dict): A dictionary containing the row data.

        Returns:
            ImageInstance: An instance of the ImageInstance class.
        """
        return cls(
            index=row["index"],
            sample_id=row["sample_id"],
            image=Image.from_row(row),
            ground_truth=GroundTruth.from_row(row),
        )
