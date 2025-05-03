from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.label import Label


class ImageInstanceBase(BaseDataInstance):
    image_file_path: str
    label: Label | None = None


class ImageInstanceCreate(ImageInstanceBase):
    dataset_split_id: SerializableUUID


class ImageInstanceUpdate(OptionalModel):
    label: Label | None = None


class ImageInstance(ImageInstanceBase, BaseDatabaseSchema):
    dataset_split_id: SerializableUUID
