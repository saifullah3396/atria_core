import enum
from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

from atria_core.schemas.utils import SerializableDateTime, SerializableUUID

if TYPE_CHECKING:
    from atria_core.types.data_instance.base import BaseDataInstance


class OptionalModel(BaseModel):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        for field in cls.model_fields.values():
            field.annotation = field.annotation | None  # <- for valid JsonSchema
            field.default = None

        cls.model_rebuild(force=True)


class DataInstanceType(str, enum.Enum):
    msgpack_object = "msgpack_object"
    image = "image_instance"
    document = "document_instance"

    @classmethod
    def from_data_model(
        self, data_model: Type["BaseDataInstance"]
    ) -> "DataInstanceType":
        """
        Convert a data model to a DataInstanceType.
        Args:
            data_model (Type[BaseDataInstance]): The data model class.
        Returns:
            DataInstanceType: The corresponding DataInstanceType.
        """
        from atria_core.types.data_instance.document import (
            DocumentInstance as AtriaDocumentInstance,
        )
        from atria_core.types.data_instance.image import (
            ImageInstance as AtriaImageInstance,
        )

        if issubclass(data_model, AtriaDocumentInstance):
            return DataInstanceType.document
        elif issubclass(data_model, AtriaImageInstance):
            return DataInstanceType.image
        else:
            raise ValueError(f"Unsupported data model: {data_model.__name__}")


class BaseDatabaseSchema(BaseModel):
    id: SerializableUUID
    created_at: SerializableDateTime
    updated_at: SerializableDateTime
