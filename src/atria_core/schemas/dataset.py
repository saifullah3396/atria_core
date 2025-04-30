from enum import Enum

from atria_core.schemas.config import Config
from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr, SerializableUUID
from atria_core.types.datasets.metadata import DatasetMetadata
from atria_core.types.datasets.splits import DatasetSplit  # noqa


class UploadStatus(str, Enum):
    UNINITIATED = "uninitiated"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class ProcessingStatus(str, Enum):
    UNINITIATED = "uninitiated"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


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
    tag: str = "default"
    total_shard_count: int | None = None
    uploaded_shard_count: int | None = None
    upload_status: UploadStatus = UploadStatus.UNINITIATED
    processing_status: ProcessingStatus = ProcessingStatus.UNINITIATED
    metadata: DatasetMetadata | None = None


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


class DatasetDeleteRequest(BaseModel):
    dataset_version: DatasetVersion


class DatasetCreateRequest(BaseModel):
    dataset_name: str
    dataset_description: str = ""
    dataset_tag: str
    is_public: bool = False
    metadata: DatasetMetadata | None = None
    data_instance_type: DataInstanceType | None = None


class DatasetCreateResponse(BaseModel):
    dataset_version: DatasetVersion


class DatasetDeleteRequest(BaseModel):
    dataset_version: DatasetVersion
