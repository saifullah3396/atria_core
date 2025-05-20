from typing import List, Optional

from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema
from atria_core.schemas.config import Config, ConfigTypes
from atria_core.schemas.utils import NameStr


class ModelBase(BaseModel):
    name: NameStr
    description: Optional[str] = None
    is_public: bool = Field(default=False)
    model_uri: str | None = Field(default=None)


class ModelCreate(ModelBase):
    pass


class ModelUpdate(BaseModel):
    name: NameStr | None = None
    description: str | None = None
    is_public: bool | None = None


class Model(ModelBase, BaseDatabaseSchema):
    configs: List[Config]

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
