from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from atria_core.types.base.data_model import TensorDataModel

if TYPE_CHECKING:
    from atria_core.types.data_instance._raw.base import BaseDataInstance  # noqa

T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound="BaseDataInstance")


class TensorBaseDataInstance(
    TensorDataModel[T_BaseDataInstance], Generic[T_BaseDataInstance]
):
    _raw_model = "atria_core.types.data_instance._raw.base.BaseDataInstance"
    index: torch.Tensor
    sample_id: str

    @property
    def key(self) -> str:
        """
        Generates a unique key for the data instance.

        The key is a combination of the UUID and the index (if present).

        Returns:
            str: The unique key for the data instance.
        """
        return str(self.sample_id.replace(".", "_"))
