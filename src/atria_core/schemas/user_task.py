import enum
import json
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import SerializableUUID


class TrainerRequest(BaseModel):
    dataset_version_id: uuid.UUID
    model_version_id: uuid.UUID
    trainer_config: dict


class InferenceRequest(BaseModel):
    instance_ids: Optional[List[SerializableUUID]] = None
    split_id: SerializableUUID
    model_id: SerializableUUID
    inference_config: Config | None = None


class DatasetProcessingRequestTypes(str, enum.Enum):
    process_shards = "process_shards"
    prepare_shards = "prepare_shards"


class UserTaskType(str, enum.Enum):
    dataset_processing = "dataset_processing"
    training = "trainer"
    inference = "inferencer"


class UserTaskStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    success = "success"
    failed = "failed"


class UserTaskBase(BaseModel):
    type: UserTaskType
    status: UserTaskStatus = Field(default=UserTaskStatus.pending)
    error_message: str | None = None
    payload: str

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value: any):
        if isinstance(value, dict):
            return json.dumps(value)
        return value


class UserTaskCreate(UserTaskBase):
    pass


class UserTaskUpdate(OptionalModel):
    status: UserTaskStatus = Field(default=UserTaskStatus.pending)
    error_message: str | None = None


class UserTask(UserTaskBase, BaseDatabaseSchema):
    user_id: SerializableUUID
    dataset_id: SerializableUUID | None = None
