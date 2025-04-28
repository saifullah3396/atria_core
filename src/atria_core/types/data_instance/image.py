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

from atria_core.types.data_instance.base import (
    BaseDataInstance,
    BatchedBaseDataInstance,
)
from atria_core.types.generic.image import BatchedImage, Image
from atria_core.types.generic.label import BatchedLabel, Label


class ImageInstance(BaseDataInstance):
    """
    A class for representing a single image with associated metadata and annotations.

    This class extends the `BaseDataInstance` class to include fields for the image data
    and its corresponding label.

    Attributes:
        image (Image): The image data associated with the instance.
        label (Label): The label associated with the image.
    """

    image: Image
    label: Label

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedImageInstance":
        """
        Constructs a batch of ImageInstance objects from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments containing lists of attributes.

        Returns:
            BatchedImageInstance: A BatchedImageInstance object containing lists of attributes.
        """
        return BatchedImageInstance(**kwargs)


class BatchedImageInstance(BatchedBaseDataInstance):
    """
    A class for representing a batch of image instances.

    This class extends the `BaseDataInstance` class to include fields for a batch of
    images and their corresponding labels.

    Attributes:
        image (BatchedImage): A batch of image data associated with the instances.
        label (BatchedLabel): A batch of labels associated with the images.
    """

    image: BatchedImage
    label: BatchedLabel
