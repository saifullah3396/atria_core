from typing import TYPE_CHECKING

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.typing.common import (
    IntField,
    ListIntField,
    ListStrField,
    StrField,
)

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.label import TensorLabel, TensorLabelList  # noqa


class Label(RawDataModel["TensorLabel"]):
    _tensor_model = "atria_core.types.generic._tensor.label.TensorLabel"
    value: IntField
    name: StrField


class LabelList(RawDataModel["TensorLabelList"]):
    _tensor_model = "atria_core.types.generic._tensor.label.TensorLabelList"
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
