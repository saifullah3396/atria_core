from typing import Self

from pydantic import model_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import (
    IntField,
    ListIntField,
    ListStrField,
    StrField,
)


class Label(BaseDataModel):
    value: IntField
    name: StrField


class LabelList(BaseDataModel):
    value: ListIntField
    name: ListStrField

    @classmethod
    def from_list(cls, labels: list[Label]) -> "LabelList":
        """
        Create a LabelList from a list of Label objects.
        """
        print([label.value for label in labels])
        return cls(
            value=[label.value for label in labels],
            name=[label.name for label in labels],
        )

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        assert len(self.value) == len(self.name), (
            "The length of 'value' and 'name' must match"
        )
        return self
