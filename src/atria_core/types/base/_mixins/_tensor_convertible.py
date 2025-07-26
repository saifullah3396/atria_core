from typing import Self

from pydantic import BaseModel


class TensorConvertible(BaseModel):
    """
    A mixin class for converting models to and from tensor-compatible versions immutably.
    """

    def to_tensor(self) -> Self:
        from atria_core.utilities.tensors import _convert_to_tensor

        updated_fields = {}

        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)

            try:
                if isinstance(value, TensorConvertible):
                    updated_fields[field_name] = value.to_tensor()
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], TensorConvertible)
                ):
                    raise RuntimeError(
                        f"Field '{field_name}' contains list of TensorConvertible, which is not supported."
                    )
                else:
                    updated_fields[field_name] = _convert_to_tensor(value)
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to tensor"
                ) from e

        return self.model_copy(update=updated_fields)

    def to_raw(self) -> Self:
        from atria_core.utilities.tensors import _convert_from_tensor

        updated_fields = {}

        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)

            try:
                if isinstance(value, TensorConvertible):
                    updated_fields[field_name] = value.to_raw()
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], TensorConvertible)
                ):
                    raise RuntimeError(
                        f"Field '{field_name}' contains list of TensorConvertible, which is not supported."
                    )
                else:
                    updated_fields[field_name] = _convert_from_tensor(value)
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to raw format"
                ) from e

        return self.model_copy(update=updated_fields)
