import json
from typing import ClassVar

from atria_core.types.base.data_model import (
    BaseDataModel,
    BaseDataModelConfigDict,
    RowSerializable,
)
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

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["word_confs", "word_angles"],
    )

    words: list[str] | None = None
    word_bboxes: list[BoundingBox] | None = None
    word_confs: list[float] | None = None
    word_angles: list[float] | None = None


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

    words: list[str] | None = None
    word_bboxes: list[BoundingBox] | None = None
    word_labels: list[Label] | None = None
    segment_level_bboxes: list[BoundingBox] | None = None


class QuestionAnswerGT(BaseDataModel):
    qa_pair: QuestionAnswerPair
    words: list[str]


class VisualQuestionAnswerGT(BaseDataModel):
    qa_pair: QuestionAnswerPair
    words: list[str]
    word_bboxes: list[BoundingBox]
    segment_level_bboxes: list[BoundingBox]


class LayoutAnalysisGT(BaseDataModel):
    annotated_objects: list[AnnotatedObject] | None = None
    words: list[str] | None = None
    word_bboxes: list[BoundingBox] | None = None


class GroundTruth(BaseDataModel, RowSerializable):
    _row_name: ClassVar[str | None] = "gt"
    _row_serialization_types: ClassVar[dict[str, str]] = {
        "classification": str,
        "ser": str,
        "ocr": str,
        "qa": str,
        "vqa": str,
        "layout": str,
    }

    classification: ClassificationGT | None = None
    ser: SERGT | None = None
    ocr: OCRGT | None = None
    qa: QuestionAnswerGT | None = None
    vqa: VisualQuestionAnswerGT | None = None
    layout: LayoutAnalysisGT | None = None

    @classmethod
    def from_url_dict(cls, url_dict: dict) -> "GroundTruth":
        import json

        values = {}
        for key, file_path in url_dict.items():
            if key.startswith("gt_"):
                import requests

                response = requests.get(file_path)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to read object from S3: {response.status_code}"
                    )
                gt_object = json.loads(response.content.decode("utf-8"))
                values[key.replace("gt_", "")] = gt_object
        return cls(**values)

    def to_row(self) -> dict:
        data = self.model_dump()
        return {
            f"gt_{key}": (json.dumps(value) if value is not None else "")
            for key, value in data.items()
        }

    @classmethod
    def from_row(cls, row: dict) -> "GroundTruth":
        return cls(
            **{
                key.replace("gt_", ""): (
                    json.loads(value) if value is not None else None
                )
                for key, value in row.items()
                if key.startswith("gt_") and value.strip() != ""
            }
        )
