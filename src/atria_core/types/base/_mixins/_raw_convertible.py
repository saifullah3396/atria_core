from typing import Any, ClassVar, Generic

from pydantic import BaseModel

from atria_core.types.base._mixins._utils import _recursive_apply
from atria_core.types.base.types import T_RawModel


class RawConvertible(BaseModel, Generic[T_RawModel]):
    """
    A mixin class for converting models to their raw data representation.
    """

    _raw_model: ClassVar[str]
    _cached_raw_model: ClassVar[Any]

    @classmethod
    def raw_data_model(cls) -> type[T_RawModel]:
        import importlib

        if not hasattr(cls, "_cached_raw_model"):
            try:
                module_name, class_name = cls._raw_model.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls._cached_raw_model = getattr(module, class_name)
            except ValueError as e:
                raise ValueError(
                    f"Invalid raw model path '{cls._raw_model}': {e}"
                ) from e
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Could not import module '{module_name}': {e}"
                ) from e
            except AttributeError as e:
                raise AttributeError(
                    f"Class '{class_name}' not found in module '{module_name}': {e}"
                ) from e
        return cls._cached_raw_model

    def to_raw(self) -> T_RawModel:
        """
        Converts the current object and its fields to tensor representations.

        Returns:
            T_RawModel: An instance of the tensor model.
        """

        from atria_core.utilities.tensors import _convert_from_tensor

        raw_model_class = self.raw_data_model()
        apply_results = _recursive_apply(self, RawConvertible, _convert_from_tensor)
        return raw_model_class.model_validate(
            apply_results, context={"no_validation": True}
        )  # type: ignore[return-value]
