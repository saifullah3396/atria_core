from atria_core.schemas.base import BaseDataInstanceStorageSchema
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.datasets.splits import DatasetSplitType


class ImageInstanceBase(BaseDataInstance):
    branch_name: str
    split: DatasetSplitType
    sample_id: str


class ImageInstance(ImageInstanceBase, BaseDataInstanceStorageSchema):
    def get_storage_instance(self):
        from atriax import storage

        return storage.image_instance
