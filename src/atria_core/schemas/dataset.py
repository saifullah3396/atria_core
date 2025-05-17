from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr, SerializableUUID
from atria_core.types.datasets.config import AtriaHubDatasetConfig
from atria_core.types.datasets.metadata import DatasetLabels, DatasetMetadata  # noqa
from atria_core.types.datasets.splits import DatasetSplitType


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


# ShardFile
class ShardFileBase(BaseModel):
    index: int
    url: str


class ShardFileCreate(ShardFileBase):
    dataset_split_id: SerializableUUID


class ShardFileUpdate(ShardFileBase, OptionalModel):
    pass


class ShardFile(ShardFileBase, BaseDatabaseSchema):
    dataset_split_id: SerializableUUID


# DatasetSplit
class DatasetSplitBase(BaseModel):
    name: DatasetSplitType
    total_shard_count: int = 0
    uploaded_shard_count: int = 0
    upload_status: UploadStatus = UploadStatus.UNINITIATED
    processing_status: ProcessingStatus = ProcessingStatus.UNINITIATED
    storage_path: str | None = None


class DatasetSplitCreate(DatasetSplitBase):
    dataset_version_id: SerializableUUID


class DatasetSplitUpdate(OptionalModel):
    # allow updating uploaded shard count and upload status
    uploaded_shard_count: int = 0
    upload_status: UploadStatus = UploadStatus.UNINITIATED
    processing_status: ProcessingStatus = ProcessingStatus.UNINITIATED


class DatasetSplit(DatasetSplitBase, BaseDatabaseSchema):
    dataset_version_id: SerializableUUID
    dataset_version: Optional["DatasetVersion"] = None
    shard_file: List[ShardFile] = Field(default_factory=list)


# DatasetVersion
class DatasetVersionBase(BaseModel):
    version: int = Field(default=0)
    config_name: NameStr = "default"
    metadata: DatasetMetadata | None = None


class DatasetVersionCreate(DatasetVersionBase):
    dataset_id: SerializableUUID


class DatasetVersionUpdate(OptionalModel):
    # allow updating version tag and version metadata
    config_name: NameStr | None = None
    metadata: DatasetMetadata | None = None


class DatasetVersion(DatasetVersionBase, BaseDatabaseSchema):
    dataset_id: SerializableUUID
    dataset: Optional["Dataset"] = None
    config: List[Config] | None = None
    dataset_split: List[DatasetSplit] = Field(default_factory=list)


# Dataset
class DatasetBase(BaseModel):
    name: NameStr
    is_public: bool = False
    data_instance_type: DataInstanceType


class DatasetCreate(DatasetBase):
    # dataset_stuff
    user_id: SerializableUUID


class DatasetUpdate(OptionalModel):
    # allow updating name of dataset and to make it public/private
    name: NameStr | None = None
    is_public: bool | None = None


class Dataset(DatasetBase, BaseDatabaseSchema):
    user_id: SerializableUUID
    versions: list["DatasetVersion"] = Field(default_factory=list)


class DatasetUploadRequest(BaseModel):
    # dataset stuff
    is_public: bool = False
    data_instance_type: DataInstanceType | None = None

    # dataset version stuff
    config: dict | None = None
    metadata: DatasetMetadata | None = None

    # dataset split stuff
    shard_index: int
    total_shard_count: int
    dataset_split_type: DatasetSplitType


class DatasetUploadResponse(BaseModel):
    dataset_split: DatasetSplit
    token: str | None = None


class DatasetDownloadRequest(BaseModel):
    name: str
    config_name: str
    username: str | None = None


class DatasetDownloadResponse(BaseModel):
    dataset_version: DatasetVersion
    access_credentials: dict | None = None


class DatasetCreateRequest(DatasetBase):
    # dataset_stuff
    user_id: SerializableUUID

    # dataset_version_stuff
    config_name: NameStr
    config: dict | None
    metadata: DatasetMetadata | None = None

    @model_validator(mode="after")
    def validate_config(self):
        if self.config is None or len(self.config) == 0:
            self.config = AtriaHubDatasetConfig(
                name=self.name, config_name=self.config_name
            )
        return self
