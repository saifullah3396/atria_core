"""
Document Instance Module

This module defines the `DocumentInstance` and `BatchedDocumentInstance` classes, which extend the `BaseDataInstance` class
to represent documents with associated metadata and annotations. These classes support fields such as images, OCR data,
labels, question-answer pairs, and annotated objects. Validation logic ensures that at least one of the required fields
(image or OCR) is provided.

Classes:
    - DocumentInstance: A class for representing a single document with associated metadata and annotations.
    - BatchedDocumentInstance: A class for representing a batch of document instances.

Dependencies:
    - pydantic: For data validation and serialization.
    - atria_core.data_types.data_instance.base: For the base data instance class.
    - atria_core.data_types.generic.annotated_object: For annotated object structures.
    - atria_core.data_types.generic.image: For image structures.
    - atria_core.data_types.generic.label: For label structures.
    - atria_core.data_types.generic.ocr: For OCR structures.
    - atria_core.data_types.generic.qa: For question-answer pair structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from pydantic import model_validator

from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image
from atria_core.types.generic.ocr import OCR


class DocumentInstance(BaseDataInstance):
    """
    A class for representing a single document with associated metadata and annotations.

    This class extends the `BaseDataInstance` class to include fields for images, OCR data,
    labels, question-answer pairs, and annotated objects. It includes validation logic to
    ensure that at least one of the required fields (image or OCR) is provided.

    Attributes:
        image (Image | None): The image data associated with the document instance. Defaults to None.
        ocr (OCR | None): The OCR data associated with the document instance. Defaults to None.
    """

    image: Image | None = None
    ocr: OCR | None = None
    ground_truth: GroundTruth = GroundTruth()

    @model_validator(mode="after")
    def validate_fields(self) -> "DocumentInstance":
        """
        Validates the fields of the `DocumentInstance`.

        Ensures that at least one of the `image` or `ocr` fields is provided.

        Returns:
            DocumentInstance: The validated instance.

        Raises:
            ValueError: If both `image` and `ocr` are None.
        """
        if self.image is None and self.ocr is None:
            raise ValueError("At least one of image or ocr must be provided")
        return self
