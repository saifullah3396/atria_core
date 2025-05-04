import enum
import uuid

from pydantic import BaseModel, Field

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableDateTime, SerializableUUID


class AgentType(str, enum.Enum):
    TRAINER = "trainer"
    EVALUATOR = "evaluator"
    INFERENCER = "inferencer"


class AgentTaskStatus(str, enum.Enum):
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class TrainerRequest(BaseModel):
    trainer_config: dict


class EvaluatorRequest(BaseModel):
    evaluator_config: dict


class InferencerRequest(BaseModel):
    instances: list[uuid.UUID]
    inference_config: dict


class AgentTaskBase(BaseModel):
    agent_type: AgentType
    status: AgentTaskStatus = Field(default=AgentTaskStatus.PENDING)
    error_message: str | None = None
    finished_at: SerializableDateTime | None = None
    request: TrainerRequest | EvaluatorRequest | InferencerRequest


class AgentTaskCreate(AgentTaskBase):
    user_id: SerializableUUID


class AgentTaskUpdate(OptionalModel):
    status: AgentTaskStatus = Field(default=AgentTaskStatus.PENDING)
    error_message: str | None = None
    finished_at: SerializableDateTime | None = None


class AgentTask(AgentTaskBase, BaseDatabaseSchema):
    user_id: SerializableUUID
