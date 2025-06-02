from enum import Enum
from typing import List

from fastapi import HTTPException
from httpx import AsyncClient

from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr, SerializableUUID
from atria_core.types.datasets.metadata import DatasetLabels, DatasetMetadata  # noqa
from atria_core.types.datasets.splits import DatasetSplitType
from pydantic import BaseModel, Field, computed_field


class ShardFileType(str, Enum):
    tar: str = "tar"


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


class DatasetStatus(str, Enum):
    READY = "ready"
    FAILED = "failed"
    PROCESSING = "processing"


class DatasetDownloadStatus(str, Enum):
    EMPTY_DATASET = "empty_dataset"
    NOT_PREPARED = "not_prepared"
    PREPARING = "preparing"
    READY = "ready"
    DATA_MODIFIED = "data_modified"


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

    async def download(self) -> bytes:
        async with AsyncClient() as client:
            response = await client.get(self.url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to load model from {self.url}: {response.text}",
            )
        return response.content


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

    @computed_field
    def size(self) -> int:
        return sum(split.filesize for split in self.shard_files)

    async def download(self) -> bytes:
        shard_files = []
        for shard_file in self.shard_files:
            try:
                content = await shard_file.download()
                return shard_files.append(content)
            except HTTPException as e:
                raise HTTPException(
                    status_code=e.status_code,
                    detail=f"Failed to download shard file {shard_file.index}: {e.detail}",
                )
        return shard_files


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

    @computed_field
    def size(self) -> int:
        return sum(split.size for split in self.splits)

    @computed_field
    def status(self) -> DatasetStatus:
        for split in self.splits:
            if (
                split.upload_status == UploadStatus.PENDING
                or split.processing_status
                in [
                    ProcessingStatus.REQUESTED,
                    ProcessingStatus.PENDING,
                ]
            ):
                return DatasetStatus.PROCESSING
            if (
                split.processing_status == ProcessingStatus.FAILED
                or split.upload_status == UploadStatus.FAILED
            ):
                return DatasetStatus.FAILED
            return DatasetStatus.READY

    @computed_field
    def download_status(self) -> DatasetDownloadStatus:
        if len(self.splits) == 0:
            return DatasetDownloadStatus.NOT_AVAILABLE
        for split in self.splits:
            if (
                split.upload_status == UploadStatus.PENDING
                or split.processing_status
                in [
                    ProcessingStatus.REQUESTED,
                    ProcessingStatus.PENDING,
                ]
            ):
                return DatasetStatus.PROCESSING
            if (
                split.processing_status == ProcessingStatus.FAILED
                or split.upload_status == UploadStatus.FAILED
            ):
                return DatasetStatus.FAILED
            return DatasetStatus.READY

    async def download(self) -> bytes:
        splits = []
        for split in self.splits:
            try:
                content = await split.download()
                splits.append(content)
            except HTTPException as e:
                raise HTTPException(
                    status_code=e.status_code,
                    detail=f"Failed to download split {split.name}: {e.detail}",
                )
        return splits


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
