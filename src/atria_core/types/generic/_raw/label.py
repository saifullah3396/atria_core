from typing import TYPE_CHECKING

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.typing.common import IntField, StrField

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.label import TensorLabel  # noqa


class Label(RawDataModel["TensorLabel"]):
    _tensor_model = "atria_core.types.generic._tensor.label.TensorLabel"
    value: IntField
    name: StrField
