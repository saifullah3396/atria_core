from typing import List, Optional

from fastapi import HTTPException
from httpx import AsyncClient
from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema
from atria_core.schemas.config import Config, ConfigTypes
from atria_core.schemas.utils import NameStr
from atria_core.types.tasks import TaskType


class ModelBase(BaseModel):
    name: NameStr
    task_type: TaskType
    description: Optional[str] = None
    is_public: bool = Field(default=False)
    model_uri: str | None = Field(default=None)
    model_size: int = Field(
        default=0,
        description="Size of the model in bytes. If not provided, it will be calculated on download.",
    )
    framework: str = Field(
        default="PyTorch",
        description="Framework used for the model (e.g., PyTorch, TensorFlow).",
    )


class ModelUpdate(BaseModel):
    name: NameStr | None = None
    description: str | None = None
    is_public: bool | None = None


class Model(ModelBase, BaseDatabaseSchema):
    configs: List[Config]

    async def download(self) -> bytes:
        async with AsyncClient() as client:
            response = await client.get(self.model_uri)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to load model from {self.model_uri}: {response.text}",
            )
        return response.content

    @property
    def inference_config(self) -> Optional[Config]:
        for config in self.configs:
            if (
                config.type == ConfigTypes.task_pipeline.value
                and config.name.startswith("inferencer/")
            ):
                return config
        return None

    @property
    def config(self) -> Optional[Config]:
        for config in self.configs:
            if config.type == ConfigTypes.model.value:
                return config
        return None
