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
        return cls(
            value=[label.value for label in labels],
            name=[label.name for label in labels],
        )
