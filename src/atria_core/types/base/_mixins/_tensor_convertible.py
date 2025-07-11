from typing import Any, ClassVar, Generic

from pydantic import BaseModel

from atria_core.types.base._mixins._utils import _recursive_apply
from atria_core.types.base.types import T_TensorModel


class TensorConvertible(BaseModel, Generic[T_TensorModel]):
    """
    A mixin class for converting models to their tensor data representation.
    """

    _tensor_model: ClassVar[str]
    _cached_tensor_model: ClassVar[Any]

    @classmethod
    def tensor_data_model(cls) -> type[T_TensorModel]:
        import importlib

        try:
            import torch  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "PyTorch is required for tensor operations but is not installed. "
                "Please install it atria with torch dependency."
                "uv add atria-core --extra torch-cpu # or torch-gpu"
            )

        if not hasattr(cls, "_cached_tensor_model"):
            try:
                module_name, class_name = cls._tensor_model.rsplit(".", 1)
            except ValueError as e:
                raise ValueError(
                    f"Invalid tensor model path '{cls._tensor_model}'. "
                    "Expected format: 'module.path.ClassName'"
                ) from e

            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(
                    f"Could not import module '{module_name}' for tensor model"
                ) from e

            try:
                cls._cached_tensor_model = getattr(module, class_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class '{class_name}' not found in module '{module_name}'"
                ) from e

        cls.model_rebuild()
        return cls._cached_tensor_model

    def to_tensor(self) -> T_TensorModel:
        """
        Converts the current object and its fields to tensor representations.

        Returns:
            T_TensorModel: An instance of the tensor model.
        """

        from atria_core.utilities.tensors import _convert_to_tensor

        tensor_model_class = self.tensor_data_model()
        apply_results = _recursive_apply(self, TensorConvertible, _convert_to_tensor)  # type: ignore[return-value]
        return tensor_model_class.model_validate(apply_results)
