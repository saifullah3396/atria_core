from pydantic import BaseModel

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableUUID


class ParamBase(BaseModel):
    key: str
    value: str


class ParamCreate(ParamBase):
    run_id: SerializableUUID


class ParamUpdate(ParamBase, OptionalModel):
    pass


class Param(ParamBase, BaseDatabaseSchema):
    run_id: SerializableUUID
