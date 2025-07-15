from typing import Self

from pydantic import BaseModel


class TensorConvertible(BaseModel):
    """
    A mixin class for converting models to their tensor data representation and back.
    """

    def to_tensor(self):
        self._to_tensor()
        return self

    def _to_tensor(self) -> None:
        from atria_core.utilities.tensors import _convert_to_tensor

        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if isinstance(field_value, TensorConvertible):
                    field_value.to_tensor()
                elif (
                    isinstance(field_value, list)
                    and len(field_value) > 0
                    and isinstance(field_value[0], TensorConvertible)
                ):
                    raise RuntimeError(
                        f"Field '{field_name}' contains list of TensorConvertible, which is not supported."
                    )
                else:
                    setattr(self, field_name, _convert_to_tensor(field_value))
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to tensor"
                ) from e

    def to_raw(self) -> Self:
        self._to_raw()
        return self

    def _to_raw(self) -> None:
        from atria_core.utilities.tensors import _convert_from_tensor

        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if isinstance(field_value, TensorConvertible):
                    field_value.to_raw()
                elif (
                    isinstance(field_value, list)
                    and len(field_value) > 0
                    and isinstance(field_value[0], TensorConvertible)
                ):
                    raise RuntimeError(
                        f"Field '{field_name}' contains list of TensorConvertible, which is not supported."
                    )
                else:
                    setattr(self, field_name, _convert_from_tensor(field_value))
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to tensor"
                ) from e
