"""
Document Instance Module

This module defines the `DocumentInstance` and `DocumentInstance` classes, which extend the `BaseDataInstance` class
to represent documents with associated metadata and annotations. These classes support fields such as images, OCR data,
labels, question-answer pairs, and annotated objects. Validation logic ensures that at least one of the required fields
(image or OCR) is provided.

Classes:
    - DocumentInstance: A class for representing a single document with associated metadata and annotations.
    - DocumentInstance: A class for representing a batch of document instances.

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

from typing import ClassVar

from atria_core.types.base.data_model import BaseDataModelConfigDict, RowSerializable
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image
from atria_core.types.generic.ocr import OCR, OCRType
from atria_core.types.ocr_parsers.hocr_parser import OCRProcessor
from pydantic import model_validator


class DocumentInstance(BaseDataInstance, RowSerializable):
    """
    A class for representing a single document with associated metadata and annotations.

    This class extends the `BaseDataInstance` class to include fields for images, OCR data,
    labels, question-answer pairs, and annotated objects. It includes validation logic to
    ensure that at least one of the required fields (image or OCR) is provided.

    Attributes:
        image (Image | None): The image data associated with the document instance. Defaults to None.
        ocr (OCR | None): The OCR data associated with the document instance. Defaults to None.
    """

    _row_name: ClassVar[str | None] = None
    _row_serialization_types: ClassVar[dict[str, str]] = {
        **BaseDataInstance.row_serialization_types(),
        **Image.row_serialization_types(),
        **GroundTruth.row_serialization_types(),
        **OCR.row_serialization_types(),
    }

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["ocr", "page_id", "total_num_pages"],
        row_serialization_types={},
    )

    page_id: int = 0
    total_num_pages: int = 1
    image: Image | None = None
    ocr: OCR
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

        if self.ocr is not None and self.ground_truth.ocr is None:
            if self.ocr.ocr_type == OCRType.tesseract:
                self.ground_truth.ocr = OCRProcessor.parse(
                    raw_ocr=self.ocr.raw_content,
                    ocr_type=self.ocr.ocr_type,
                )

        return self

    def to_row(self) -> dict:
        """
        Converts the instance to a row format suitable for storage or serialization.
        Returns:
            dict: A dictionary representation of the instance, with keys prefixed by "image_", "ground_truth_", and "ocr_".
        """
        row = {
            **self.model_dump(exclude={"ground_truth", "image", "ocr"}),
            **self.image.to_row(),
            **self.ground_truth.to_row(),
            **self.ocr.to_row(),
        }
        return row

    @classmethod
    def from_row(cls, row: dict) -> "DocumentInstance":
        return cls(
            index=row["index"],
            sample_id=row["sample_id"],
            page_id=row["page_id"],
            total_num_pages=row["total_num_pages"],
            image=Image.from_row(row),
            ground_truth=GroundTruth.from_row(row),
            ocr=OCR.from_row(row),
        )
