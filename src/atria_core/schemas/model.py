from typing import List, Optional

from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema
from atria_core.schemas.config import Config
from atria_core.schemas.utils import NameStr


class ModelBase(BaseModel):
    name: NameStr
    description: Optional[str] = None
    is_public: bool = Field(default=False)
    version: int = Field(default=0)
    version_tag: str = Field(default_factory="v0")
    model_uri: str


class ModelCreate(ModelBase):
    configs: List[Config]

    # @property
    # def inference_config(self) -> Optional[Config]:
    #     for config in self.configs:
    #         if (
    #             config.type == ConfigTypes.task_pipeline.value
    #             and config.name.startswith("inferencer/")
    #         ):
    #             return config
    #     return None

    # @property
    # def model_config(self) -> Optional[Config]:
    #     for config in self.configs:
    #         if config.type == ConfigTypes.model.value:
    #             return config
    #     return None


class ModelUpdate(BaseModel):
    name: NameStr | None = None
    description: str | None = None
    version_tag: str | None = None
    is_public: bool | None = None


class Model(ModelBase, BaseDatabaseSchema):
    configs: List[Config] = Field(default_factory=list)


# class ModelDownloadRequest(BaseModel):
#     name: str | None = None
#     version_tag: str | None = None
#     username: str | None = None
#     model_version_id: uuid.UUID | None = None

#     @model_validator(mode="after")
#     @classmethod
#     def validate_model_download_request(cls, values):
#         if not values.get("model_version_id") and not values.get("name"):
#             raise ValueError("Either model_version_id or name must be provided.")
#         if not values.get("version_tag") and not values.get("model_version_id"):
#             raise ValueError("Either version_tag or model_version_id must be provided.")
#         return values


# class ModelDownloadResponse(BaseModel):
#     download_url: str


# class ModelCreateRequest(BaseModel):
#     version_tag: str
#     description: str
#     is_public: bool = False
#     inference_config: dict

#     @field_validator("inference_config", mode="before")
#     @classmethod
#     def parse_config(cls, value):
#         if isinstance(value, str):
#             import json

#             return json.loads(value)
#         return value
