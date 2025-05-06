import enum
import uuid

from pydantic import BaseModel, computed_field

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import (
    NameStr,
    SerializableUUID,
    _convert_dict_to_schema,
    _generate_hash_from_dict,
)


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
    data: dict

    @computed_field
    @property
    def schema_hash(self) -> str:
        return _generate_hash_from_dict(_convert_dict_to_schema(self.data))

    @computed_field
    @property
    def hash(self) -> str:
        return _generate_hash_from_dict(self.data)


class ConfigCreate(ConfigBase):
    user_id: SerializableUUID


class ConfigUpdate(ConfigBase, OptionalModel):
    pass


class Config(ConfigBase, BaseDatabaseSchema):
    user_id: uuid.UUID


class ConfigFilter(BaseModel):
    type: NameStr | None = None
    name: NameStr | None = None
