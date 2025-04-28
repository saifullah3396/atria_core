from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.dataset import DatasetSplit
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.document import (
    DocumentInstance as AtriaDocumentInstance,
)


class DocumentInstanceBase(AtriaDocumentInstance):
    split: DatasetSplit
    data_instance_type: DataInstanceType


class DocumentInstanceCreate(DocumentInstanceBase):
    dataset_version_id: SerializableUUID


class DocumentInstanceUpdate(DocumentInstanceBase, OptionalModel):
    pass


class DocumentInstance(DocumentInstanceBase, BaseDatabaseSchema):
    dataset_version_id: SerializableUUID
