from typing import TYPE_CHECKING, Self

import torch
from pydantic import model_validator

from atria_core.types.base.data_model import TensorDataModel

if TYPE_CHECKING:
    from atria_core.types.generic._raw.label import Label, LabelList  # noqa


class TensorLabel(TensorDataModel["Label"]):
    _raw_model = "atria_core.types.generic._raw.label.Label"
    value: torch.Tensor
    name: str

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self._is_batched:
            assert self.value.ndim == 1, "Expected a 1D tensor for batched labels"
        else:
            assert self.value.ndim == 0, "Expected a scalar tensor"
        return self


class TensorLabelList(TensorDataModel["LabelList"]):
    _raw_model = "atria_core.types.generic._raw.label.LabelList"
    value: torch.Tensor
    name: list[str]

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self._is_batched:
            assert self.value.ndim == 2, "Expected a 2D tensor for batched labels"
        else:
            assert self.value.ndim == 1, "Expected a 1D tensor for list of labels"
        return self
