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

from typing import Any

from pydantic import field_serializer, field_validator, model_validator

from atria_core.types.data_instance.base import (
    BaseDataInstance,
    BatchedBaseDataInstance,
)
from atria_core.types.generic.annotated_object import (
    AnnotatedObjectSequence,
    BatchedAnnotatedObjectSequence,
)
from atria_core.types.generic.image import BatchedImage, Image
from atria_core.types.generic.label import BatchedLabel, Label
from atria_core.types.generic.ocr import OCR, BatchedOCR
from atria_core.types.generic.question_answer_pair import (
    BatchedQuestionAnswerPairSequence,
    QuestionAnswerPairSequence,
)


class DocumentInstance(BaseDataInstance):
    """
    A class for representing a single document with associated metadata and annotations.

    This class extends the `BaseDataInstance` class to include fields for images, OCR data,
    labels, question-answer pairs, and annotated objects. It includes validation logic to
    ensure that at least one of the required fields (image or OCR) is provided.

    Attributes:
        image (Image | None): The image data associated with the document instance. Defaults to None.
        ocr (OCR | None): The OCR data associated with the document instance. Defaults to None.
        label (Label | None): The label associated with the document instance. Defaults to None.
        question_answer_pairs (List[QuestionAnswerPair] | None): A sequence of question-answer pairs
            associated with the document instance. Defaults to None.
        annotated_objects (List[AnnotatedObject] | None): A sequence of annotated objects
            associated with the document instance. Defaults to None.
    """

    image: Image | None = None
    ocr: OCR | None = None
    label: Label | None = None
    question_answer_pairs: QuestionAnswerPairSequence | None = None
    annotated_objects: AnnotatedObjectSequence | None = None

    @field_validator("question_answer_pairs")
    @classmethod
    def validate_question_answer_pairs(cls, value: Any) -> list[dict]:
        """
        Validates the `question_answer_pairs` field.
        If the value is a list, it converts it to a `QuestionAnswerPairSequence`.
        Args:
            cls: The class being validated.
            value: The value to validate.
        Returns:
            QuestionAnswerPairSequence: The validated question-answer pairs.
        """
        if isinstance(value, list):
            return QuestionAnswerPairSequence.from_list(value)
        return value

    @field_validator("annotated_objects")
    @classmethod
    def validate_annotated_objects(cls, value: Any) -> list[dict]:
        """
        Validates the `annotated_objects` field.
        If the value is a list, it converts it to a `AnnotatedObjectSequence`.
        Args:
            cls: The class being validated.
            value: The value to validate.
        Returns:
            AnnotatedObjectSequence: The validated annotated objects.
        """
        if isinstance(value, list):
            return AnnotatedObjectSequence.from_list(value)
        return value

    @field_serializer("question_answer_pairs")
    def serialize_question_answer_pairs(
        self, value: QuestionAnswerPairSequence
    ) -> list[dict]:
        """
        Serializes the `question_answer_pairs` field to a list of dictionaries.

        Args:
            value (QuestionAnswerPairSequence): The question-answer pairs to serialize.

        Returns:
            list[dict]: A list of dictionaries representing the serialized question-answer pairs.
        """
        if value is None:
            return value
        return [v.model_dump() for v in value.to_list()]

    @field_serializer("annotated_objects")
    def serialize_annotated_objects(self, value: AnnotatedObjectSequence) -> list[dict]:
        """
        Serializes the `annotated_objects` field to a list of dictionaries.

        Args:
            value (AnnotatedObjectSequence): The annotated objects to serialize.

        Returns:
            list[dict]: A list of dictionaries representing the serialized annotated objects.
        """
        if value is None:
            return value
        return [v.model_dump() for v in value.to_list()]

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedDocumentInstance":
        """
        Constructs a batch of DocumentInstance objects from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments containing lists of attributes.

        Returns:
            BatchedDocumentInstance: A BatchedDocumentInstance object containing lists of attributes.
        """
        return BatchedDocumentInstance(**kwargs)

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


class BatchedDocumentInstance(BatchedBaseDataInstance):
    """
    A class for representing a batch of document instances.

    This class extends the `BaseDataInstance` class to include fields for a batch of
    images, OCR data, labels, question-answer pairs, and annotated objects.

    Attributes:
        image (BatchedImage | None): A batch of image data associated with the document instances. Defaults to None.
        ocr (BatchedOCR | None): A batch of OCR data associated with the document instances. Defaults to None.
        label (BatchedLabel | None): A batch of labels associated with the document instances. Defaults to None.
        question_answer_pairs (BatchedSequenceQuestionAnswerPairs | None): A batch of question-answer pairs
            associated with the document instances. Defaults to None.
        annotated_objects (BatchedSequenceAnnotatedObjects | None): A batch of annotated objects
            associated with the document instances. Defaults to None.
    """

    image: BatchedImage | None = None
    ocr: BatchedOCR | None = None
    label: BatchedLabel | None = None
    question_answer_pairs: BatchedQuestionAnswerPairSequence | None = None
    annotated_objects: BatchedAnnotatedObjectSequence | None = None
