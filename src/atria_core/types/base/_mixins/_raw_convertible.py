from typing import Any, ClassVar, Generic

from pydantic import BaseModel

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

    def to_raw(self) -> "T_RawModel":
        """
        Converts the model instance to its raw data representation.

        Returns:
            T_RawModel: The raw data model instance.

        Raises:
            ValueError: If a field cannot be converted.
        """
        from atria_core.utilities.tensors import _convert_from_tensor

        raw_model_class = self.raw_data_model()
        raw_fields: dict[str, Any] = {}
        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if isinstance(field_value, RawConvertible):
                    raw_fields[field_name] = field_value.to_raw()
                elif (
                    isinstance(field_value, list)
                    and len(field_value) > 0
                    and isinstance(field_value[0], RawConvertible)
                ):

                    def safe_to_raw(item: Any):
                        if isinstance(item, RawConvertible):
                            return item.to_raw()
                        else:
                            raise RuntimeError(
                                "A list with variable types is not supported."
                            )

                    raw_fields[field_name] = [safe_to_raw(item) for item in field_value]
                else:
                    raw_fields[field_name] = _convert_from_tensor(field_value)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert field '{field_name}' to raw format."
                ) from e

        return raw_model_class(**raw_fields)
