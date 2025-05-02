import uuid

from pydantic import BaseModel, model_validator

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import (
    NameStr,
    SerializableUUID,
    _convert_dict_to_schema,
    _generate_hash_from_dict,
)


class ConfigBase(BaseModel):
    type: NameStr  # e.g. "engine", "dataset", etc.
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


class ConfigDependencyBase(BaseModel):
    ref_type: str  # e.g. "engine_step"
    ref_default_value: str | None = None


class ConfigDependencyCreate(ConfigDependencyBase):
    config_id: SerializableUUID


class ConfigDependencyUpdate(ConfigDependencyBase, OptionalModel):
    pass


class ConfigDependency(ConfigDependencyBase, BaseDatabaseSchema):
    config_id: SerializableUUID
