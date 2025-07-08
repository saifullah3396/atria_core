from typing import Any, ClassVar, Generic

from pydantic import BaseModel

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

        return cls._cached_tensor_model

    def to_tensor(self) -> T_TensorModel:
        """
        Converts the current object and its fields to tensor representations.

        Returns:
            T_TensorModel: An instance of the tensor model.
        """

        from atria_core.utilities.tensors import _convert_to_tensor

        tensor_model_class = self.tensor_data_model()
        tensor_fields = {}
        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if isinstance(field_value, TensorConvertible):
                    tensor_fields[field_name] = field_value.to_tensor()
                elif (
                    isinstance(field_value, list)
                    and len(field_value) > 0
                    and isinstance(field_value[0], TensorConvertible)
                ):

                    def safe_to_tensor(item: Any):
                        if isinstance(item, TensorConvertible):
                            return item.to_tensor()
                        else:
                            raise RuntimeError(
                                "A list with variable types is not supported."
                            )

                    tensor_fields[field_name] = [
                        safe_to_tensor(item) for item in field_value
                    ]
                else:
                    tensor_fields[field_name] = _convert_to_tensor(field_value)  # type: ignore[assignment]
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to tensor"
                ) from e
        tensor_model_class.model_rebuild()
        return tensor_model_class(**tensor_fields)
