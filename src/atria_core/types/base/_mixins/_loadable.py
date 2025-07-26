from typing import Self

from pydantic import BaseModel


class Loadable(BaseModel):
    """
    A mixin class for managing the loading and unloading of model fields.
    """

    def load(self) -> Self:
        updated_fields = {}

        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Loadable):
                updated_fields[field_name] = value.load()

        updated_fields.update(self._load())
        return self.model_copy(update=updated_fields)

    def unload(self) -> Self:
        updated_fields = {}

        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Loadable):
                updated_fields[field_name] = value.unload()

        updated_fields.update(self._unload())
        return self.model_copy(update=updated_fields)

    def _load(self) -> None:
        """
        Placeholder for custom load logic.
        """
        return {}

    def _unload(self) -> None:
        """
        Placeholder for custom unload logic.
        """
        return {}
