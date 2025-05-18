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

import json
from atria_core.types.base.data_model import BaseDataModelConfigDict
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image
from atria_core.types.generic.ocr import OCR, OCRType
from atria_core.types.ocr_parsers.hocr_parser import OCRProcessor
from atria_core.schemas.data_instances.document_instance import (
    DocumentInstance as DocumentInstanceSchema,
)
from pydantic import model_validator


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

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["ocr", "page_id", "total_num_pages"],
    )

    doc_id: str
    page_id: int = 0
    total_num_pages: int = 1
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

        if self.ocr is not None and self.ground_truth.ocr is None:
            if self.ocr.ocr_type == OCRType.TESSERACT:
                self.ground_truth.ocr = OCRProcessor.parse(
                    raw_ocr=self.ocr.raw_content,
                    ocr_type=self.ocr.ocr_type,
                )

        return self

    @classmethod
    def from_schema(cls, schema: DocumentInstanceSchema):
        ocr, image = None, None
        if "image_file_path" in schema.data:
            image = Image(file_path=schema.data["image_file_path"])
        if "ocr_file_path" in schema.data:
            ocr = OCR(file_path=schema.data["ocr_file_path"], ocr_type=schema.ocr_type)

        ground_truth = GroundTruth()
        for key in ground_truth.__dict__.keys():
            if f"gt_{key}" in schema.data:
                with open(schema.data[f"gt_{key}"], "r") as f:
                    json.loads(f.read())
                setattr(ground_truth, key, schema.data[key])
        return DocumentInstance(
            id=schema.id,
            index=schema.index,
            doc_id=schema.doc_id,
            page_id=schema.page_id,
            total_num_pages=schema.total_num_pages,
            image=image,
            ocr=ocr,
            ground_truth=ground_truth,
        )
