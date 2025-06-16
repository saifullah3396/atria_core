import enum

from pydantic import BaseModel, ConfigDict, Field

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import NameStr, SerializableDateTime, SerializableUUID


class RunStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    success = "success"
    failed = "failed"


class RunBase(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    name: NameStr = Field(..., min_length=1, max_length=255)
    status: RunStatus = Field(default=RunStatus.pending)
    error_message: str | None = None
    finished_at: SerializableDateTime | None = None


class RunCreate(RunBase):
    experiment_id: SerializableUUID


class RunUpdate(RunBase, OptionalModel):
    pass


class Run(RunBase, BaseDatabaseSchema):
    experiment_id: SerializableUUID
