from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth


class ImageInstanceBase(BaseDataInstance):
    index: int
    sample_path: str | None = None
    data: dict


class ImageInstanceCreate(ImageInstanceBase):
    split_id: SerializableUUID


class ImageInstanceUpdate(OptionalModel):
    ground_truth: GroundTruth


class ImageInstance(ImageInstanceBase, BaseDatabaseSchema):
    split_id: SerializableUUID
