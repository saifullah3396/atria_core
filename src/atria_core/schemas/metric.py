from pydantic import BaseModel

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableUUID


class MetricBase(BaseModel):
    key: str
    value: float
    step: int


class MetricCreate(MetricBase):
    user_id: SerializableUUID
    run_id: SerializableUUID


class MetricUpdate(MetricBase, OptionalModel):
    pass


class Metric(MetricBase, BaseDatabaseSchema):
    run_id: SerializableUUID
