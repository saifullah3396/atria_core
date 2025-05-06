from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.annotated_object import (
    AnnotatedObject,
)
from typing import List, Optional

from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.label import Label
from atria_core.types.generic.question_answer_pair import (
    QuestionAnswerPair,
)


class ClassificationGT(BaseDataModel):
    label: Label


class OCRGroundTruth(BaseDataModel):
    """
    A class for representing processed OCR data.

    This class provides functionality for managing processed OCR data, including
    the original content and the parsed graph representation.

    Attributes:
        words (List[str]): A list of words extracted from the OCR content.
        word_bboxes (List[BoundingBox]): A list of bounding boxes for each word.
        word_confs (List[float]): A list of confidence scores for each word.
        word_angles (List[float]): A list of angles for each word.
    """

    words: List[str] | None = None
    word_bboxes: List[BoundingBox] | None = None
    word_confs: List[float] | None = None
    word_angles: List[float] | None = None


class NERGroundTruth(BaseDataModel):
    """
    A class for representing processed bounding box and label data.

    This class provides functionality for managing processed OCR data, including
    the original content and the parsed graph representation.

    Attributes:
        words (List[str]): A list of words extracted from the OCR content.
        word_bboxes (List[BoundingBox]): A list of bounding boxes for each word.
        word_labels (List[Label]): A list of labels for each word.
    """

    words: List[str] | None = None
    word_bboxes: List[BoundingBox] | None = None
    word_labels: List[Label] | None = None


class QuestionAnswerGT(BaseDataModel):
    pairs: List[QuestionAnswerPair]


class LayoutAnalysisGT(BaseDataModel):
    objects: List[AnnotatedObject] | None = None


class GroundTruth(BaseDataModel):
    classification: Optional[ClassificationGT] = None
    ner: Optional[NERGroundTruth] = None
    ocr: Optional[OCRGroundTruth] = None
    qa: Optional[QuestionAnswerGT] = None
    layout: Optional[LayoutAnalysisGT] = None
