from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from atria_core.utilities.repr import RepresentationMixin


class DataTransform(RepresentationMixin):
    def __init__(self, input_path: str | None = None):
        self.input_path = input_path

    @abstractmethod
    def _apply_transforms(self, input: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _apply_transforms method."
        )

    def _validate_and_apply_transforms(
        self, input: Any | Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self.input_path is not None:
            attrs = self.input_path.split(".")
            obj = input
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            current_attr = getattr(obj, attrs[-1])
            assert current_attr is not None, (
                f"{self.__class__.__name__} transform requires {self.input_path} to be present in the sample."
            )
            setattr(obj, attrs[-1], self._apply_transforms(current_attr))
            return input
        else:
            return self._apply_transforms(input)

    def __call__(
        self, input: Any | Mapping[str, Any] | list[Mapping[str, Any]]
    ) -> Any | Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(input, list):
            return [self(s) for s in input]
        return self._validate_and_apply_transforms(input)


class DataTransformsDict(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="forbid"
    )

    train: DataTransform | OrderedDict[str, DataTransform] | None = None
    evaluation: DataTransform | OrderedDict[str, DataTransform] | None = None

    def compose(self, type: str) -> DataTransform | Callable:
        from torchvision.transforms import Compose  # type: ignore

        tf = getattr(self, type, None)
        if tf is None:
            raise ValueError(
                f"Transformations for type '{type}' are not defined in {self.__class__.__name__}."
            )
        if isinstance(tf, dict):
            return Compose(list(tf.values()))
        return tf


class Compose(RepresentationMixin):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input
