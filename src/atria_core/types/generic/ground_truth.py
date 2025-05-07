from typing import List, Optional

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.annotated_object import AnnotatedObject
from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.label import Label
from atria_core.types.generic.question_answer_pair import QuestionAnswerPair


class ClassificationGT(BaseDataModel):
    label: Label


class OCRGT(BaseDataModel):
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


class SERGT(BaseDataModel):
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
    segment_level_bboxes: List[BoundingBox] | None = None


class QuestionAnswerGT(BaseDataModel):
    qa_pairs: List[QuestionAnswerPair]


class VisualQuestionAnswerGT(BaseDataModel):
    qa_pairs: List[QuestionAnswerPair]
    words: list[str]
    word_bboxes: list[BoundingBox]
    segment_level_bboxes: list[BoundingBox]


class LayoutAnalysisGT(BaseDataModel):
    annotated_objects: List[AnnotatedObject] | None = None
    words: List[str] | None = None
    word_bboxes: List[BoundingBox] | None = None


class GroundTruth(BaseDataModel):
    classification: Optional[ClassificationGT] = None
    ser: Optional[SERGT] = None
    ocr: Optional[OCRGT] = None
    qa: Optional[QuestionAnswerGT] = None
    vqa: Optional[VisualQuestionAnswerGT] = None
    layout: Optional[LayoutAnalysisGT] = None
