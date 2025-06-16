import enum
import uuid

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, computed_field

from atria_core.schemas.base import BaseStorageDatabaseSchema
from atria_core.schemas.utils import NameStr, _generate_hash_from_dict


class ConfigTypes(str, enum.Enum):
    batch_sampler = "batch_sampler"
    data_pipeline = "data_pipeline"
    dataset = "dataset"
    data_transform = "data_transform"
    dataset_splitter = "dataset_splitter"
    dataset_storage_manager = "dataset_storage_manager"
    engine = "engine"
    engine_step = "engine_step"
    lr_scheduler_factory = "lr_scheduler_factory"
    metric_factory = "metric_factory"
    model = "model"
    model_pipeline = "model_pipeline"
    optimizer_factory = "optimizer_factory"
    task_pipeline = "task_pipeline"


class ConfigBase(BaseModel):
    type: ConfigTypes
    name: NameStr
    is_public: bool = False


class ConfigCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    type: ConfigTypes
    name: NameStr
    data: dict
    is_public: bool = False

    @property
    def hash(self) -> str:
        return _generate_hash_from_dict(self.data)


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: NameStr = None
    data: dict = None
    is_public: bool = None


class Config(ConfigBase, BaseStorageDatabaseSchema):
    user_id: uuid.UUID

    @computed_field
    @property
    def data_url(self) -> str | None:
        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == "data"
                ),
                None,
            )
        return None

    async def fetch_data(self) -> DictConfig:
        return OmegaConf.create(yaml.safe_load(await self.fetch_object(self.data_url)))
