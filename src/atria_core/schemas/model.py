from fastapi import File, UploadFile
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, computed_field

from atria_core.schemas.base import BaseStorageDatabaseSchema
from atria_core.schemas.utils import NameStr, SerializableUUID
from atria_core.types.tasks import ModelType, TaskType


class ModelBase(BaseModel):
    name: NameStr
    description: str
    task_type: TaskType
    is_public: bool = Field(default=False)


class ModelExternalImport(BaseModel):
    name: NameStr
    task: TaskType
    model_type: ModelType
    model_name: str
    description: str | None = None
    is_public: bool = False
    ckpt_state_path: str = "state_dict"
    model_state_path: str = "_model"
    model_file: UploadFile = File(None)
    card_file: UploadFile = File(None)


class ModelUpload(BaseModel):
    # task type is assigned from checkpoint
    name: NameStr
    description: str
    is_public: bool = False
    model_file: UploadFile = File(...)
    card_file: UploadFile = File(None)


class ModelUpdate(BaseModel):
    name: NameStr | None = None
    description: str | None = None
    is_public: bool | None = None


class ModelListItem(ModelBase, BaseStorageDatabaseSchema):
    user_id: SerializableUUID


class Model(ModelBase, BaseStorageDatabaseSchema):
    user_id: SerializableUUID

    @computed_field
    @property
    def model_url(self) -> str | None:
        from atriax import storage

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == storage.model.__default_model_path__
                ),
                None,
            )
        return None

    @computed_field
    @property
    def card_url(self) -> str | None:
        from atriax import storage

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == storage.model.__default_card_path__
                ),
                None,
            )
        return None

    @computed_field
    @property
    def model_config_url(self) -> str | None:
        from atriax import storage

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == storage.model.__default_model_config_path__
                ),
                None,
            )
        return None

    @computed_field
    @property
    def inference_config_url(self) -> str | None:
        from atriax import storage

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key == storage.model.__default_inference_config_path__
                ),
                None,
            )
        return None

    async def fetch_model(self) -> DictConfig:
        return await self.fetch_object(self.model_url)

    async def fetch_card(self) -> str:
        return (await self.fetch_object(self.card_url)).decode("utf-8")

    async def fetch_model_config(self) -> str:
        return OmegaConf.create(
            yaml.safe_load(await self.fetch_object(self.model_config_url))
        )

    async def fetch_inference_config(self) -> str:
        return OmegaConf.create(
            yaml.safe_load(await self.fetch_object(self.inference_config_url))
        )
