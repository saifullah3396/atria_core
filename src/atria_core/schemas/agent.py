import uuid

from pydantic import BaseModel


class InferencRequest(BaseModel):
    model_id: uuid.UUID
    instances: list[uuid.UUID]
    inference_config: dict
