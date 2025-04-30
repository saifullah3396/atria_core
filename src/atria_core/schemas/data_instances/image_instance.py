from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.datasets.splits import DatasetSplit
from atria_core.types.generic.label import Label


class ImageInstanceBase(BaseDataInstance):
    image_file_path: str
    label: Label
    split: DatasetSplit
    data_instance_type: DataInstanceType


class ImageInstanceCreate(ImageInstanceBase):
    dataset_version_id: SerializableUUID


class ImageInstanceUpdate(ImageInstanceBase, OptionalModel):
    pass


class ImageInstance(ImageInstanceBase, BaseDatabaseSchema):
    dataset_version_id: SerializableUUID
