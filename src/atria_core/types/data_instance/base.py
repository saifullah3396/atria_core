import uuid
from typing import Annotated

from pydantic import Field, field_serializer, field_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.annotations import Annotation, AnnotationType
from atria_core.types.typing.common import OptIntField, StrField, TableSchemaMetadata


class BaseDataInstance(BaseDataModel):
    _tensor_data_model = "atria_core.types.data_instance._tensor.base.BaseDataInstance"
    index: OptIntField = None
    sample_id: StrField = Field(default_factory=lambda: str(uuid.uuid4()))
    annotations: Annotated[
        list[Annotation] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @property
    def key(self) -> str:
        """
        Generates a unique key for the data instance.

        The key is a combination of the UUID and the index (if present).

        Returns:
            str: The unique key for the data instance.
        """
        return str(self.sample_id.replace(".", "_"))

    def get_annotation_by_type(
        self, annotation_type: AnnotationType
    ) -> Annotation | None:
        """
        Retrieves the first annotation of the specified type.

        Args:
            annotation_type (AnnotationType): The type of annotation to retrieve.

        Returns:
            Annotation | None: The first annotation of the specified type, or None if not found.
        """
        if self.annotations is None:
            return None
        for annotation in self.annotations:
            if annotation.type == annotation_type:
                return annotation
        return None

    @field_validator("annotations", mode="before")
    def validate_annotations(cls, value) -> list[Annotation] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
                assert isinstance(value, list), "Annotations JSON must be a list"

                annotations = []
                for item in value:
                    assert isinstance(item, dict), "Each annotation must be a dict"

                    ann_type = item.pop("type", None)
                    annotations.append(Annotation.from_type(ann_type, item))
                return annotations
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}")

        return value

    @field_serializer("annotations")
    def serialize_annotations(self, value: list[Annotation]) -> str | None:
        import json

        if value is not None:
            return json.dumps(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
            )
        return None
