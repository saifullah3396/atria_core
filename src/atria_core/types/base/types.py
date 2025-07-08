from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from atria_core.types.base.data_model import RawDataModel, TensorDataModel

T_RawModel = TypeVar("T_RawModel", bound="RawDataModel")
T_TensorModel = TypeVar("T_TensorModel", bound="TensorDataModel")
