import enum
from typing import Annotated

from pydantic import field_serializer, field_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.annotated_object import AnnotatedObjectList
from atria_core.types.generic.label import Label, LabelList
from atria_core.types.generic.question_answer_pair import (
    ExtractiveQAPair,
    GenerativeQAItem,
)
from atria_core.types.typing.common import TableSchemaMetadata


class AnnotationType(str, enum.Enum):
    classification = "classification"
    entity_labeling = "entity_labeling"
    extractive_qa = "extractive_qa"
    generative_qa = "generative_qa"
    layout = "layout"


class Annotation(BaseDataModel):
    _type: AnnotationType

    @classmethod
    def from_type(cls, annotation_type: AnnotationType, params: dict) -> "Annotation":
        if annotation_type == AnnotationType.classification:
            return ClassificationAnnotation(**params)
        elif annotation_type == AnnotationType.entity_labeling:
            return EntityLabelingAnnotation(**params)
        elif annotation_type == AnnotationType.extractive_qa:
            return ExtractiveQAAnnotation(**params)
        elif annotation_type == AnnotationType.generative_qa:
            return GenerativeQAAnnotation(**params)
        elif annotation_type == AnnotationType.layout:
            return LayoutAnalysisAnnotation(**params)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")

    def model_dump(self, *args, **kwargs):
        return {**super().model_dump(*args, **kwargs), "type": self._type}


class ClassificationAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.classification
    label: Label


class EntityLabelingAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.entity_labeling
    word_labels: LabelList


class ExtractiveQAAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.extractive_qa
    qa_pairs: Annotated[
        list[ExtractiveQAPair] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @field_validator("qa_pairs", mode="before")
    def validate_qa_pairs(cls, value) -> list[ExtractiveQAPair] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}")
        return value

    @field_serializer("qa_pairs")
    def serialize_qa_pairs(self, value: list[ExtractiveQAPair]) -> str | None:
        import json

        if value is not None:
            return json.dumps(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
            )
        return None


class LayoutAnalysisAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.layout
    annotated_objects: Annotated[
        AnnotatedObjectList | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @field_validator("annotated_objects", mode="before")
    def validate_annotated_objects(cls, value) -> AnnotatedObjectList | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}")
        return value

    @field_serializer("annotated_objects")
    def serialize_annotated_objects(self, value: AnnotatedObjectList) -> str | None:
        import json

        if value is not None:
            return json.dumps(value.model_dump())
        return None


class GenerativeQAAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.generative_qa
    qa_pairs: Annotated[
        list[GenerativeQAItem] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @field_validator("qa_pairs", mode="before")
    def validate_qa_pairs(cls, value) -> list[GenerativeQAItem] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}")
        return value

    @field_serializer("qa_pairs")
    def serialize_qa_pairs(self, value: list[GenerativeQAItem]) -> str | None:
        import json

        if value is not None:
            return json.dumps(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
            )
        return None
