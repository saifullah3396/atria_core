from typing import Self

from pydantic import BaseModel, PrivateAttr


class Loadable(BaseModel):
    """
    A mixin class for managing the loading and unloading of model fields.
    """

    _is_loaded = PrivateAttr(default=False)

    def load(self) -> Self:
        """
        Loads all fields of the model, with enhanced logging and error handling.

        Returns:
            Self: The instance with fields loaded.

        Raises:
            Exception: If loading fails for any field.
        """
        if not self._is_loaded:
            for field_name in self.__class__.model_fields:
                field_value = getattr(self, field_name)
                if isinstance(field_value, Loadable):
                    field_value.load()
            self._load()
            self._is_loaded = True
        return self

    def unload(self) -> Self:
        """
        Unloads all fields of the model, with enhanced logging and error handling.

        Returns:
            Self: The instance with fields unloaded.

        Raises:
            Exception: If unloading fails for any field.
        """
        if self._is_loaded:
            for field_name in self.__class__.model_fields:
                field_value = getattr(self, field_name)
                if isinstance(field_value, Loadable):
                    field_value.unload()
            self._unload()
            self._is_loaded = False
        return self

    def _load(self) -> None:
        """
        Placeholder for custom load logic.
        """

    def _unload(self) -> None:
        """
        Placeholder for custom unload logic.
        """
