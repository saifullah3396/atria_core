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

import json
from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image
from atria_core.schemas.data_instances.image_instance import (
    ImageInstance as ImageInstanceSchema,
)


class ImageInstance(BaseDataInstance):
    """
    A class for representing a single image with associated metadata and annotations.

    This class extends the `BaseDataInstance` class to include fields for the image data
    and its corresponding label.

    Attributes:
        image (Image): The image data associated with the instance.
        ground_truth (GroundTruth): The ground truth associated with the image.
    """

    image: Image
    ground_truth: GroundTruth = GroundTruth()

    @classmethod
    def from_schema(cls, schema: ImageInstanceSchema):
        image = None, None
        if "image_file_path" in schema.data:
            image = Image(file_path=schema.data["image_file_path"])

        ground_truth = GroundTruth()
        for key in ground_truth.__dict__.keys():
            if f"gt_{key}" in schema.data:
                with open(schema.data[f"gt_{key}"], "r") as f:
                    json.loads(f.read())
                setattr(ground_truth, key, schema.data[key])
        return ImageInstance(
            id=schema.id,
            index=schema.index,
            image=image,
            ground_truth=ground_truth,
        )
