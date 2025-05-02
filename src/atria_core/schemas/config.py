import enum
import uuid

from pydantic import BaseModel, model_validator

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
    schema_hash: str
    hash: str
    data: dict

    @model_validator(mode="before")
    @classmethod
    def validate_and_generate(cls, values):
        data = values.get("data")
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")

        if len(data) == 0:
            raise ValueError("data cannot be an empty dictionary")

        # Generate schema_hash
        values["schema_hash"] = _generate_hash_from_dict(_convert_dict_to_schema(data))

        # Generate version if not provided
        values["hash"] = _generate_hash_from_dict(data)
        return values


class ConfigCreate(ConfigBase):
    user_id: SerializableUUID


class ConfigUpdate(ConfigBase, OptionalModel):
    pass


class Config(ConfigBase, BaseDatabaseSchema):
    user_id: uuid.UUID


class ConfigFilter(BaseModel):
    type: NameStr | None = None
    name: NameStr | None = None
