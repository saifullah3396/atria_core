from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.image import ImageInstance as AtriaImageInstance
from atria_core.types.datasets.splits import DatasetSplit


class ImageInstanceBase(AtriaImageInstance):
    split: DatasetSplit
    data_instance_type: DataInstanceType


class ImageInstanceCreate(ImageInstanceBase):
    dataset_version_id: SerializableUUID


class ImageInstanceUpdate(ImageInstanceBase, OptionalModel):
    pass


class ImageInstance(ImageInstanceBase, BaseDatabaseSchema):
    dataset_version_id: SerializableUUID
