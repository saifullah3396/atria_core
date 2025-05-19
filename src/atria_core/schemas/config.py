import enum
import uuid

from pydantic import BaseModel

from atria_core.schemas.base import BaseDatabaseSchema
from atria_core.schemas.utils import NameStr, SerializableUUID, _generate_hash_from_dict


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
    path: str
    is_public: bool = False

    @property
    def hash(self) -> str:
        return _generate_hash_from_dict(self.data)


class ConfigCreate(BaseModel):
    type: ConfigTypes
    name: NameStr
    data: dict
    is_public: bool = False
    user_id: SerializableUUID

    @property
    def hash(self) -> str:
        return _generate_hash_from_dict(self.data)


class ConfigUpdate(BaseModel):
    name: NameStr = None


class Config(ConfigBase, BaseDatabaseSchema):
    user_id: uuid.UUID


class ConfigFilter(BaseModel):
    type: NameStr | None = None
    name: NameStr | None = None
