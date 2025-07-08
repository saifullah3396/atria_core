from typing import TYPE_CHECKING

import torch
from pydantic import field_validator

from atria_core.types.base.data_model import TensorDataModel

if TYPE_CHECKING:
    from atria_core.types.generic._raw.label import Label  # noqa


class TensorLabel(TensorDataModel["Label"]):
    _raw_model = "atria_core.types.generic._raw.label.Label"
    value: torch.Tensor
    name: str

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: torch.Tensor) -> torch.Tensor:
        assert value.ndim in [0], "Expected a scalar tensor"
        return value
