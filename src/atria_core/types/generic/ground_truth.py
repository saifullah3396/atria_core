import json
from typing import Annotated

from pydantic import field_serializer, field_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.annotated_object import AnnotatedObjectList
from atria_core.types.generic.bounding_box import BoundingBoxList
from atria_core.types.generic.label import Label, LabelList
from atria_core.types.generic.question_answer_pair import QuestionAnswerPair
from atria_core.types.typing.common import (
    ListStrField,
    OptListFloatField,
    OptListStrField,
    TableSchemaMetadata,
)


class ClassificationGT(BaseDataModel):
    label: Label


class OCRGT(BaseDataModel):
    words: OptListStrField = None
    word_bboxes: BoundingBoxList | None = None
    word_confs: OptListFloatField = None
    word_angles: OptListFloatField = None


class SERGT(BaseDataModel):
    words: OptListStrField = None
    word_bboxes: BoundingBoxList | None = None
    word_labels: LabelList | None = None
    segment_level_bboxes: BoundingBoxList | None = None


class QuestionAnswerGT(BaseDataModel):
    qa_pair: QuestionAnswerPair
    words: ListStrField


class VisualQuestionAnswerGT(BaseDataModel):
    qa_pair: QuestionAnswerPair
    words: ListStrField
    word_bboxes: BoundingBoxList | None = None
    segment_level_bboxes: BoundingBoxList | None = None


class LayoutAnalysisGT(BaseDataModel):
    annotated_objects: AnnotatedObjectList | None = None
    words: OptListStrField = None
    word_bboxes: BoundingBoxList | None = None


class GroundTruth(BaseDataModel):
    classification: Annotated[
        ClassificationGT | None, TableSchemaMetadata(pa_type="string")
    ] = None
    ser: Annotated[SERGT | None, TableSchemaMetadata(pa_type="string")] = None
    ocr: Annotated[OCRGT | None, TableSchemaMetadata(pa_type="string")] = None
    qa: Annotated[QuestionAnswerGT | None, TableSchemaMetadata(pa_type="string")] = None
    vqa: Annotated[
        VisualQuestionAnswerGT | None, TableSchemaMetadata(pa_type="string")
    ] = None
    layout: Annotated[
        LayoutAnalysisGT | None, TableSchemaMetadata(pa_type="string")
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
    def serialize_gt(self, value) -> str | None:
        if value is not None:
            return json.dumps(value.model_dump())
        return None
