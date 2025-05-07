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
from typing import List, Optional

from pydantic import BaseModel, model_validator

from atria_core.types.base.data_model import BaseDataModel, BaseDataModelConfigDict
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

    TESSERACT = "tesseract"
    EASY_OCR = "easy_ocr"
    GOOGLE_VISION = "google_vision"
    AWS_REKOGNITION = "aws_rekognition"
    AZURE_OCR = "azure_ocr"
    CUSTOM = "custom"
    OTHER = "other"


class OCRLevel(str, Enum):
    PAGE = "page"
    BLOCK = "block"
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"


class OCRGraphNode(BaseModel):
    id: int
    word: Optional[str] = None
    level: Optional[OCRLevel] = None
    bbox: Optional[BoundingBox] = None
    segment_level_bbox: Optional[BoundingBox] = None
    conf: Optional[float] = None
    angle: Optional[float] = None
    label: Optional[Label] = None


class OCRGraphLink(BaseModel):
    source: int
    target: int
    relation: Optional[str] = None


class OCRGraph(BaseModel):
    directed: Optional[bool]
    multigraph: Optional[bool]
    graph: Optional[dict]
    nodes: List[OCRGraphNode]
    links: List[OCRGraphLink]

    @property
    def words(self) -> List[str]:
        return [node.word for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_bboxes(self) -> List[BoundingBox]:
        return [node.bbox for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_segment_level_bboxes(self) -> List[BoundingBox]:
        return [
            node.segment_level_bbox
            for node in self.nodes
            if node.level == OCRLevel.WORD
        ]

    @property
    def word_labels(self) -> List[Label]:
        return [node.label for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_confs(self) -> List[float]:
        return [node.conf for node in self.nodes if node.level == OCRLevel.WORD]

    @property
    def word_angles(self) -> List[float]:
        return [node.angle for node in self.nodes if node.level == OCRLevel.WORD]

    @classmethod
    def from_word_level_content(
        cls,
        words: List[str],
        word_bboxes: List[BoundingBox],
        word_labels: List[Label],
        word_segment_level_bboxes: List[BoundingBox] = None,
        word_confs: List[float] = None,
        word_angles: List[float] = None,
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


class OCR(BaseDataModel):
    """
    A class for managing OCR data.

    This class provides functionality for loading OCR data from files, parsing the content,
    and validating the data.

    Attributes:
        file_path (PydanticFilePath): The file path to the OCR data. Defaults to None.
        ocr_type (Optional[OCRType]): The type of OCR system used. Defaults to None.
        raw_content (Optional[str]): The raw OCR content. Defaults to None.
    """

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["file_path"],
        batch_merge_fields=["ocr_type"],
    )

    file_path: PydanticFilePath = None
    ocr_type: Optional[OCRType] = None
    raw_content: Optional[str] = None

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
            if not Path(self.file_path).exists():
                raise FileNotFoundError("Either file_path or graph must be provided.")
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.raw_content = f.read()
                if self.raw_content.startswith("b'"):
                    self.raw_content = ast.literal_eval(self.raw_content).decode(
                        "utf-8"
                    )
                assert len(self.raw_content) > 0, "OCR content is empty."
        return self
