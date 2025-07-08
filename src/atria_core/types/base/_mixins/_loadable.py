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
            try:
                for field_name in self.__class__.model_fields:
                    try:
                        field_value = getattr(self, field_name)
                        if isinstance(field_value, Loadable):
                            field_value.load()
                        elif isinstance(field_value, list):
                            if field_value and isinstance(field_value[0], Loadable):
                                for i, item in enumerate(field_value):
                                    try:
                                        item.load()
                                    except Exception as e:
                                        raise RuntimeError(
                                            f"Error loading item {i} in list '{field_name}'"
                                        ) from e
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load field '{field_name}': {str(e)}"
                        ) from e

                self._load()
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Loading failed for {self.__class__.__name__}: {str(e)}"
                ) from e

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
            try:
                for field_name in self.__class__.model_fields:
                    try:
                        field_value = getattr(self, field_name)
                        if isinstance(field_value, Loadable):
                            field_value.unload()
                        elif isinstance(field_value, list):
                            if field_value and isinstance(field_value[0], Loadable):
                                for i, item in enumerate(field_value):
                                    try:
                                        item.unload()
                                    except Exception as e:
                                        raise RuntimeError(
                                            f"Error unloading item {i} in list '{field_name}'"
                                        ) from e
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to unload field '{field_name}'"
                        ) from e

                self._unload()
                self._is_loaded = False
            except Exception as e:
                raise RuntimeError(
                    f"Loading failed for {self.__class__.__name__}"
                ) from e

        return self

    def _load(self) -> None:
        """
        Placeholder for custom load logic.
        """

    def _unload(self) -> None:
        """
        Placeholder for custom unload logic.
        """
