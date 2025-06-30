"""
OCR Module

This module defines the `OCR`, `OCRContent`, and `BatchedOCR` classes, which represent OCR data and its parsed content.
It includes functionality for loading OCR data from files, parsing the content, and validating the data.
The module also defines the `OCRType` enum for specifying the type of OCR system used.

Classes:
    - OCRType: Enum for OCR types.
    - OCRContent: A class for representing parsed OCR content, including words, bounding boxes, and confidence scores.
    - BatchedOCRContent: A class for managing batched OCR content.
    - OCR: A class for managing OCR data, including loading, parsing, and validation.
    - BatchedOCR: A class for managing batched OCR data.

Dependencies:
    - ast: For evaluating string literals.
    - enum: For defining the `OCRType` enum.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - pydantic: For data validation and serialization.
    - atria_core.utilities.encoding: For encoding and decoding string data.
    - atria_core.data_types.base.data_model: For the base data model class.
    - atria_core.data_types.generic.bounding_box: For bounding box structures.
    - atria_core.data_types.generic.label: For sequence label structures.
    - atria_core.data_types.typing.common: For common type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import ast
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, model_validator

from atria_core.types.base.data_model import (
    BaseDataModel,
    BaseDataModelConfigDict,
    RowSerializable,
)
from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.label import Label
from atria_core.types.typing.common import PydanticFilePath


class OCRType(str, Enum):
    """
    Enum for OCR types.

    Attributes:
        TESSERACT (str): Tesseract OCR.
        EASY_OCR (str): EasyOCR.
        GOOGLE_VISION (str): Google Vision OCR.
        AWS_REKOGNITION (str): AWS Rekognition OCR.
        AZURE_OCR (str): Azure OCR.
        OTHER (str): Custom OCR implementation.
    """

    tesseract = "tesseract"
    easy_ocr = "easy_ocr"
    google_vision = "google_vision"
    aws_rekognition = "aws_rekognition"
    azure_ocr = "azure_ocr"
    custom = "custom"
    other = "other"


class OCRLevel(str, Enum):
    PAGE = "page"
    BLOCK = "block"
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"


class OCRGraphNode(BaseModel):
    id: int
    word: str | None = None
    level: OCRLevel | None = None
    bbox: BoundingBox | None = None
    segment_level_bbox: BoundingBox | None = None
    conf: float | None = None
    angle: float | None = None
    label: Label | None = None


class OCRGraphLink(BaseModel):
    source: int
    target: int
    relation: str | None = None


class OCRGraph(BaseModel):
    directed: bool | None
    multigraph: bool | None
    graph: dict | None
    nodes: list[OCRGraphNode]
    links: list[OCRGraphLink]

    @property
    def words(self) -> list[str]:
        return [node.word for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_bboxes(self) -> list[BoundingBox]:
        return [node.bbox for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_segment_level_bboxes(self) -> list[BoundingBox]:
        return [
            node.segment_level_bbox
            for node in self.nodes
            if node.level == OCRLevel.WORD
        ]

    @property
    def word_labels(self) -> list[Label]:
        return [node.label for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_confs(self) -> list[float]:
        return [node.conf for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_angles(self) -> list[float]:
        return [node.angle for node in self.nodes if node.level == OCRLevel.WORD]

    @classmethod
    def from_word_level_content(
        cls,
        words: list[str],
        word_bboxes: list[BoundingBox],
        word_labels: list[Label],
        word_segment_level_bboxes: list[BoundingBox] = None,
        word_confs: list[float] = None,
        word_angles: list[float] = None,
    ) -> "OCRGraph":
        nodes = []
        for i, word in enumerate(words):
            node = OCRGraphNode(
                id=i,
                word=word,
                level=OCRLevel.WORD,
                bbox=word_bboxes[i] if word_bboxes is not None else None,
                segment_level_bbox=(
                    word_segment_level_bboxes[i]
                    if word_segment_level_bboxes is not None
                    else None
                ),
                conf=word_confs[i] if word_confs is not None else None,
                angle=word_angles[i] if word_angles is not None else None,
                label=word_labels[i] if word_labels is not None else None,
            )
            nodes.append(node)

        links = []
        for i in range(len(words) - 1):
            links.append(
                OCRGraphLink(
                    source=i,
                    target=i + 1,
                    relation="next_word",
                )
            )

        return OCRGraph(
            directed=True, multigraph=False, graph={}, nodes=nodes, links=links
        )


class OCR(BaseDataModel, RowSerializable):
    """
    A class for managing OCR data.

    This class provides functionality for loading OCR data from files, parsing the content,
    and validating the data.

    Attributes:
        file_path (PydanticFilePath): The file path to the OCR data. Defaults to None.
        type (Optional[OCRType]): The type of OCR system used. Defaults to None.
        raw_content (Optional[str]): The raw OCR content. Defaults to None.
    """

    _row_name: ClassVar[str | None] = "ocr"
    _row_serialization_types: ClassVar[dict[str, str]] = {
        "file_path": str,
        "type": str,
        "raw_content": str,
    }

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["file_path"],
        batch_merge_fields=["type"],
    )

    file_path: PydanticFilePath = None
    type: OCRType | None = None
    raw_content: str | None = None

    @model_validator(mode="after")
    def validate(self):
        """
        Validates the OCR data.

        Ensures that at least one of `file_path`, `content`, or `parsed_content` is provided.

        Returns:
            OCR: The validated OCR instance.

        Raises:
            ValueError: If none of the required fields are provided.
        """
        if self.raw_content is None:
            if self.file_path is None:
                raise ValueError("Either file_path or raw_content must be provided.")
            if str(self.file_path).startswith(("http", "https")):
                import requests

                response = requests.get(self.file_path)
                if response.status_code != 200:
                    raise ValueError(f"Failed to load image from URL: {self.file_path}")
                self.raw_content = response.content.decode("utf-8")
            else:
                if not Path(self.file_path).exists():
                    raise FileNotFoundError(f"File not found: {self.file_path}")
                with open(self.file_path, encoding="utf-8") as f:
                    self.raw_content = f.read()
                    if self.raw_content.startswith("b'"):
                        self.raw_content = ast.literal_eval(self.raw_content).decode(
                            "utf-8"
                        )
                    assert len(self.raw_content) > 0, "OCR content is empty."
        return self
