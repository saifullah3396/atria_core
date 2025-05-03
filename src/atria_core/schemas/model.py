from typing import List, Optional

from codename import codename
from pydantic import BaseModel, Field, field_validator

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr, SerializableUUID


class ModelVersionBase(BaseModel):
    version: int = Field(default=0)
    version_tag: str = Field(default_factory=codename)
    model_uri: str


class ModelVersionCreate(ModelVersionBase):
    model_id: SerializableUUID
    config_id: SerializableUUID


class ModelVersionUpdate(ModelVersionBase):
    pass


class ModelVersion(ModelVersionBase, BaseDatabaseSchema):
    model_id: SerializableUUID
    config_id: SerializableUUID
    model: "Model"
    config: Config


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


class ModelUploadRequest(BaseModel):
    name: str
    description: str
    version_tag: str
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


class ModelDownloadRequest(BaseModel):
    name: str
    version_tag: str
    user_id: SerializableUUID


class ModelDownloadResponse(BaseModel):
    download_url: str
