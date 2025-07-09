from typing import Self

from pydantic import BaseModel, PrivateAttr

from atria_core.types.base._mixins._utils import _recursive_apply_in_place


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
            _recursive_apply_in_place(
                self, Loadable, lambda x: x._load() if isinstance(x, Loadable) else None
            )  # type: ignore[return-value]
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
            _recursive_apply_in_place(
                self, Loadable, lambda x: x._unload() if isinstance(x, Loadable) else x
            )  # type: ignore[return-value]
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
