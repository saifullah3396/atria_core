import json
from typing import TYPE_CHECKING, Annotated

import pyarrow as pa
from pydantic import field_serializer, field_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.generic._raw.annotated_object import AnnotatedObjectList
from atria_core.types.generic._raw.bounding_box import BoundingBoxList
from atria_core.types.generic._raw.label import Label, LabelList
from atria_core.types.generic._raw.question_answer_pair import QuestionAnswerPair
from atria_core.types.typing.common import TableSchemaMetadata

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.ground_truth import (
        TensorClassificationGT,  # noqa
        TensorGroundTruth,  # noqa
        TensorLayoutAnalysisGT,  # noqa
        TensorOCRGT,  # noqa
        TensorQuestionAnswerGT,  # noqa
        TensorSERGT,  # noqa
        TensorVisualQuestionAnswerGT,  # noqa
    )


class ClassificationGT(RawDataModel["TensorClassificationGT"]):
    _tensor_model = (
        "atria_core.types.generic._tensor.ground_truth.TensorClassificationGT"
    )
    label: Label


class OCRGT(RawDataModel["TensorOCRGT"]):
    _tensor_model = "atria_core.types.generic._tensor.ground_truth.TensorOCRGT"
    words: list[str] | None = None
    word_bboxes: BoundingBoxList | None = None
    word_confs: list[float] | None = None
    word_angles: list[float] | None = None


class SERGT(RawDataModel["TensorSERGT"]):
    _tensor_model = "atria_core.types.generic._tensor.ground_truth.TensorSERGT"
    words: list[str] | None = None
    word_bboxes: BoundingBoxList | None = None
    word_labels: LabelList | None = None
    segment_level_bboxes: BoundingBoxList | None = None


class QuestionAnswerGT(RawDataModel["TensorQuestionAnswerGT"]):
    _tensor_model = (
        "atria_core.types.generic._tensor.ground_truth.TensorQuestionAnswerGT"
    )
    qa_pair: QuestionAnswerPair
    words: list[str]


class VisualQuestionAnswerGT(RawDataModel["TensorVisualQuestionAnswerGT"]):
    _tensor_model = (
        "atria_core.types.generic._tensor.ground_truth.TensorVisualQuestionAnswerGT"
    )
    qa_pair: QuestionAnswerPair
    words: list[str]
    word_bboxes: BoundingBoxList
    segment_level_bboxes: BoundingBoxList


class LayoutAnalysisGT(RawDataModel["TensorLayoutAnalysisGT"]):
    _tensor_model = (
        "atria_core.types.generic._tensor.ground_truth.TensorLayoutAnalysisGT"
    )
    annotated_objects: AnnotatedObjectList | None = None
    words: list[str] | None = None
    word_bboxes: BoundingBoxList | None = None


class GroundTruth(RawDataModel["TensorGroundTruth"]):
    _tensor_model = "atria_core.types.generic._tensor.ground_truth.TensorGroundTruth"
    classification: Annotated[
        ClassificationGT | None, TableSchemaMetadata(pyarrow=pa.string())
    ] = None
    ser: Annotated[SERGT | None, TableSchemaMetadata(pyarrow=pa.string())] = None
    ocr: Annotated[OCRGT | None, TableSchemaMetadata(pyarrow=pa.string())] = None
    qa: Annotated[QuestionAnswerGT | None, TableSchemaMetadata(pyarrow=pa.string())] = (
        None
    )
    vqa: Annotated[
        VisualQuestionAnswerGT | None, TableSchemaMetadata(pyarrow=pa.string())
    ] = None
    layout: Annotated[
        LayoutAnalysisGT | None, TableSchemaMetadata(pyarrow=pa.string())
    ] = None

    @field_validator(
        "classification", "ser", "ocr", "qa", "vqa", "layout", mode="before"
    )
    def validate_gt(cls, value):
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}")
        return value

    @field_serializer("classification", "ser", "ocr", "qa", "vqa", "layout")
    def serialize_gt(self, value) -> str:
        if value is not None:
            return json.dumps(value.model_dump())
        return None
