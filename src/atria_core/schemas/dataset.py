from enum import Enum
from typing import List

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, computed_field

from atria_core.schemas.base import BaseStorageDatabaseSchema, DataInstanceType
from atria_core.schemas.user_task import UserTask, UserTaskStatus
from atria_core.schemas.utils import NameStr, SerializableUUID


class DatasetStatus(str, Enum):
    unavailable = "unavailable"
    available = "available"
    outdated = "outdated"


# Dataset
class DatasetBase(BaseModel):
    name: NameStr
    description: str | None = None
    is_public: bool = False
    status: DatasetStatus = DatasetStatus.unavailable
    data_instance_type: DataInstanceType


class DatasetListItem(DatasetBase, BaseStorageDatabaseSchema):
    user_id: SerializableUUID


class Dataset(DatasetBase, BaseStorageDatabaseSchema):
    user_id: SerializableUUID
    tasks: List[UserTask] = Field(default_factory=list)

    @computed_field
    @property
    def under_processing(self) -> "Dataset":
        if len(self.tasks) > 0 and any(
            task.status in {UserTaskStatus.pending, UserTaskStatus.in_progress}
            for task in self.tasks
        ):
            return True
        return False

    @computed_field
    @property
    def shard_urls(self) -> List[str]:
        from atriax import storage

        if self.storage_objects:
            shard_urls = []
            for obj in self.storage_objects:
                if obj.object_key.startswith(storage.dataset.__default_shards_path__):
                    shard_urls.append(obj.presigned_url)
            return shard_urls if shard_urls else []
        return []

    @computed_field
    @property
    def card_url(self) -> str | None:
        from atriax import storage

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == storage.dataset.__default_card_path__
                ),
                None,
            )
        return None

    @computed_field
    @property
    def config_url(self) -> str | None:
        from atriax import storage

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == storage.dataset.__default_config_path__
                ),
                None,
            )
        return None

    async def fetch_card(self) -> str:
        return await self.fetch_object(self.card_url).decode("utf-8")

    async def fetch_config(self) -> str:
        return OmegaConf.create(
            yaml.safe_load(await self.fetch_object(self.config_url))
        )


class DatasetDownloadRequest(BaseModel):
    name: str
    config_name: str
    username: str | None = None


class DatasetDownloadResponse(BaseModel):
    dataset: Dataset
    access_credentials: dict | None = None
