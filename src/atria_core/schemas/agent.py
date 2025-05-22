import enum
import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.config import Config
from atria_core.schemas.utils import SerializableUUID


class AgentType(str, enum.Enum):
    TRAINER = "trainer"
    EVALUATOR = "evaluator"
    INFERENCER = "inferencer"


class AgentTaskStatus(str, enum.Enum):
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class TrainerRequest(BaseModel):
    dataset_version_id: uuid.UUID
    model_version_id: uuid.UUID
    trainer_config: dict


class EvaluatorRequest(BaseModel):
    evaluator_config: dict


class InferenceRequest(BaseModel):
    instance_ids: List[SerializableUUID]
    split_id: SerializableUUID
    model_id: SerializableUUID
    inference_config: Config | None = None


class AgentTaskBase(BaseModel):
    agent_type: AgentType
    status: AgentTaskStatus = Field(default=AgentTaskStatus.PENDING)
    error_message: str | None = None
    finished_at: datetime | None = None
    request: TrainerRequest | EvaluatorRequest | InferenceRequest


class AgentTaskCreate(AgentTaskBase):
    pass


class AgentTaskUpdate(OptionalModel):
    status: AgentTaskStatus = Field(default=AgentTaskStatus.PENDING)
    error_message: str | None = None
    finished_at: datetime | None = None


class AgentTask(AgentTaskBase, BaseDatabaseSchema):
    user_id: SerializableUUID
