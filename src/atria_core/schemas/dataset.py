from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr, SerializableUUID
from atria_core.types.datasets.metadata import DatasetLabels, DatasetMetadata  # noqa
from atria_core.types.datasets.splits import DatasetSplitType


class UploadStatus(str, Enum):
    UNINITIATED = "uninitiated"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class ProcessingStatus(str, Enum):
    UNINITIATED = "uninitiated"
    REQUESTED = "requested"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


# ShardFile
class ShardFileBase(BaseModel):
    index: int
    url: str
    nsamples: int = 0
    filesize: int = 0


class ShardFileCreate(ShardFileBase):
    split_id: SerializableUUID


class ShardFileUpdate(ShardFileBase, OptionalModel):
    pass


class ShardFile(ShardFileBase, BaseDatabaseSchema):
    split_id: SerializableUUID


# DatasetSplit
class DatasetSplitBase(BaseModel):
    name: DatasetSplitType
    upload_status: UploadStatus = UploadStatus.UNINITIATED
    processing_status: ProcessingStatus = ProcessingStatus.UNINITIATED


class DatasetSplitCreate(DatasetSplitBase):
    dataset_id: SerializableUUID


class DatasetSplitUpdate(OptionalModel):
    # allow updating uploaded shard count and upload status
    upload_status: UploadStatus = UploadStatus.UNINITIATED
    processing_status: ProcessingStatus = ProcessingStatus.UNINITIATED


class DatasetSplit(DatasetSplitBase, BaseDatabaseSchema):
    dataset_id: SerializableUUID
    shard_files: List[ShardFile] = Field(default_factory=list)


# Dataset
class DatasetBase(BaseModel):
    name: NameStr
    config_name: NameStr = "default"
    dataset_metadata: DatasetMetadata | None = None
    is_public: bool = False
    data_instance_type: DataInstanceType


class DatasetCreate(DatasetBase):
    # on creation we also receive the config
    # this config is received from cli but from UI
    # it will be null
    config: dict | None = None


class DatasetUpdate(OptionalModel):
    # allow updating name of dataset and to make it public/private
    name: NameStr | None = None
    config_name: NameStr | None = None
    is_public: bool | None = None
    dataset_metadata: DatasetMetadata | None = None


class Dataset(DatasetBase, BaseDatabaseSchema):
    user_id: SerializableUUID
    config: Config
    splits: List[DatasetSplit] = Field(default_factory=list)


class SplitUploadRequest(BaseModel):
    shard_index: int
    total_shard_count: int


class SplitUploadResponse(BaseModel):
    dataset_split: DatasetSplit
    token: str | None = None


class DatasetDownloadRequest(BaseModel):
    name: str
    config_name: str
    username: str | None = None


class DatasetDownloadResponse(BaseModel):
    dataset: Dataset
    access_credentials: dict | None = None
