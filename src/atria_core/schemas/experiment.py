from pydantic import BaseModel, ConfigDict, Field

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import NameStr, SerializableUUID


class ExperimentBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: NameStr = Field(..., min_length=1, max_length=255)
    is_public: bool = False


class ExperimentCreate(ExperimentBase):
    pass


class ExperimentUpdate(ExperimentBase, OptionalModel):
    pass


class Experiment(ExperimentBase, BaseDatabaseSchema):
    user_id: SerializableUUID
