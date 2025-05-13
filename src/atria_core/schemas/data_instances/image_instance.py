from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.label import Label


class ImageInstanceBase(BaseDataInstance):
    sample_path: str | None = None


class ImageInstanceCreate(ImageInstanceBase):
    dataset_split_id: SerializableUUID


class ImageInstanceUpdate(OptionalModel):
    ground_truth: GroundTruth


class ImageInstance(ImageInstanceBase, BaseDatabaseSchema):
    dataset_split_id: SerializableUUID
