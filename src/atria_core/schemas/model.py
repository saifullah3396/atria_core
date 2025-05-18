from typing import List, Optional
import uuid

from codename import codename
from pydantic import BaseModel, Field, field_validator, model_validator

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr, SerializableUUID


class ModelVersionBase(BaseModel):
    version: int = Field(default=0)
    version_tag: str = Field(default_factory=codename)
    model_uri: str


class ModelVersionCreate(ModelVersionBase):
    model_id: SerializableUUID


class ModelVersionUpdate(ModelVersionBase):
    pass


class ModelVersion(ModelVersionBase, BaseDatabaseSchema):
    model_id: SerializableUUID
    model: Optional["Model"] = None


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
    version_tag: str
    description: str
    is_public: bool = False


class ModelUploadResponse(BaseModel):
    model_version: ModelVersion
    token: Optional[str] = None


class ModelDownloadRequest(BaseModel):
    name: str | None = None
    version_tag: str | None = None
    username: str | None = None
    model_version_id: uuid.UUID | None = None

    @model_validator(mode="after")
    @classmethod
    def validate_model_download_request(cls, values):
        if not values.get("model_version_id") and not values.get("name"):
            raise ValueError("Either model_version_id or name must be provided.")
        if not values.get("version_tag") and not values.get("model_version_id"):
            raise ValueError("Either version_tag or model_version_id must be provided.")
        return values


class ModelDownloadResponse(BaseModel):
    download_url: str


class ModelCreateRequest(BaseModel):
    version_tag: str
    description: str
    is_public: bool = False
    inference_config: dict

    @field_validator("inference_config", mode="before")
    @classmethod
    def parse_config(cls, value):
        if isinstance(value, str):
            import json

            return json.loads(value)
        return value
