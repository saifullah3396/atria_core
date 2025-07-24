from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict
from transformers import AutoConfig

from atria_core.utilities.repr import RepresentationMixin


class DataTransform(AutoConfig, RepresentationMixin):
    def __init__(self, apply_path: str | None = None):
        self.apply_path = apply_path

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _apply_transforms(self, input: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _apply_transforms method."
        )

    def _validate_and_apply_transforms(
        self, input: Any | Mapping[str, Any], apply_path_override: str | None = None
    ) -> Mapping[str, Any]:
        apply_path = apply_path_override or self.apply_path
        if apply_path is not None:
            attrs = apply_path.split(".")
            obj = input
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            current_attr = getattr(obj, attrs[-1])
            if current_attr is None:
                # If the attribute is None, we cannot apply the transformation
                # and should raise an error or handle it gracefully.
                raise ValueError(
                    f"You must provide a valid input for '{apply_path}' to apply the transformation '{self.name}'."
                    f"'{current_attr}' in object {obj}"
                )
            setattr(obj, attrs[-1], self._apply_transforms(current_attr))
            return input
        else:
            return self._apply_transforms(input)

    def __call__(
        self,
        input: Any | Mapping[str, Any] | list[Mapping[str, Any]],
        apply_path: str | None = None,
    ) -> Any | Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(input, list):
            return [self(s, apply_path=apply_path) for s in input]
        return self._validate_and_apply_transforms(
            input, apply_path_override=apply_path
        )


class DataTransformsDict(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="forbid"
    )

    train: DataTransform | None = None
    evaluation: DataTransform | None = None


class Compose(RepresentationMixin):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input
