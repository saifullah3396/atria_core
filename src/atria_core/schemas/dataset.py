from enum import Enum

from atria_core.schemas.config import Config
from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.utils import NameStr, SerializableUUID
from atria_core.types.datasets.metadata import DatasetLabels  # noqa
from atria_core.types.datasets.splits import DatasetSplit  # noqa


class UploadStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    FAILED = "failed"
    SUCCESS = "success"


class ProcessingStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    FAILED = "failed"
    SUCCESS = "success"


class DatasetMetadata(BaseModel):
    citation: str | None = None
    homepage: str | None = None
    license: str | None = None
    dataset_labels: DatasetLabels | None = None
    data_model: str | None = None


class DatasetBase(BaseModel):
    name: NameStr
    description: str | None
    is_public: bool = False
    data_instance_type: DataInstanceType


class DatasetCreate(DatasetBase):
    config_name: str = "default"
    user_id: SerializableUUID


class DatasetUpdate(DatasetBase, OptionalModel):
    pass


class Dataset(DatasetBase, BaseDatabaseSchema):
    user_id: SerializableUUID
    versions: list["DatasetVersion"] = Field(default_factory=list)


class DatasetVersionBase(BaseModel):
    config_name: str = "default"
    total_shard_count: int
    uploaded_shard_count: int = 0
    upload_status: UploadStatus = UploadStatus.NOT_STARTED
    processing_status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    metadata: DatasetMetadata


class DatasetVersionCreate(DatasetVersionBase):
    dataset_id: SerializableUUID
    config_id: SerializableUUID


class DatasetVersionUpdate(DatasetVersionBase, OptionalModel):
    pass


class DatasetVersion(DatasetVersionBase, BaseDatabaseSchema):
    dataset_id: SerializableUUID
    config_id: SerializableUUID
    dataset: Dataset
    config: Config


class DatasetUploadRequest(BaseModel):
    shard_index: int
    total_shard_count: int
    dataset_name: str
    dataset_config_name: str
    dataset_description: str
    dataset_split: DatasetSplit
    is_public: bool = False
    config: dict | None = None
    metadata: DatasetMetadata | None = None
    data_instance_type: DataInstanceType | None = None


class DatasetUploadResponse(BaseModel):
    dataset_version: DatasetVersion
    token: str | None = None


class DatasetRollbackUploadRequest(BaseModel):
    dataset_version: DatasetVersion
