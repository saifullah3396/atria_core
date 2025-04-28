from typing import List, Optional

from codename import codename
from pydantic import BaseModel, Field, field_validator

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import NameStr, SerializableUUID


class ModelBase(BaseModel):
    name: NameStr
    description: Optional[str] = None
    is_public: bool = Field(default=False)


class ModelCreate(ModelBase):
    user_id: SerializableUUID


class ModelUpdate(ModelBase, OptionalModel):
    pass


class Model(ModelBase, BaseDatabaseSchema):
    user_id: SerializableUUID
    versions: List["ModelVersion"] = []


class ModelVersionBase(BaseModel):
    version: int = Field(default=0)
    tag: str = Field(default_factory=lambda: codename(separator="-"))
    model_uri: str


class ModelVersionCreate(ModelVersionBase):
    model_id: SerializableUUID
    config_id: SerializableUUID | None = None


class ModelVersionUpdate(ModelVersionBase):
    pass


class ModelVersion(ModelVersionBase, BaseDatabaseSchema):
    model_id: SerializableUUID
    config_id: SerializableUUID | None = None


class ModelUploadRequest(BaseModel):
    model_name: str
    model_description: str
    model_tag: str
    is_public: bool = False
    config: Optional[dict] = None

    @field_validator("config", mode="before")
    @classmethod
    def parse_config(cls, value):
        if isinstance(value, str):
            import json

            return json.loads(value)
        return value


class ModelUploadResponse(BaseModel):
    model_version: ModelVersion
    token: Optional[str] = None


class ModelDeleteRequest(BaseModel):
    model_version: ModelVersion


class ModelDownloadRequest(BaseModel):
    model_name: str
    model_tag: str


class ModelDownloadResponse(BaseModel):
    download_url: str
