from abc import abstractmethod
from collections.abc import Mapping
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict

from atria_core.utilities.repr import RepresentationMixin


class DataTransform(BaseModel, RepresentationMixin):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="forbid"
    )

    apply_path: str | None = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @cached_property
    def config(self) -> dict:
        return self.prepare_build_config()

    def prepare_build_config(self):
        from hydra_zen import builds
        from omegaconf import OmegaConf

        init_fields = {k: getattr(self, k) for k in self.__class__.model_fields}
        cfg = builds(
            self.__class__,
            populate_full_signature=True,
            **init_fields,
        )
        return OmegaConf.to_container(OmegaConf.create(cfg))

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

    @abstractmethod
    def _apply_transforms(self, input: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _apply_transforms method."
        )


class DataTransformsDict(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="forbid"
    )

    train: DataTransform | None = None
    evaluation: DataTransform | None = None

    @property
    def config(self) -> dict:
        from hydra_zen import builds
        from omegaconf import OmegaConf

        return OmegaConf.to_container(
            OmegaConf.create(
                builds(
                    self.__class__,
                    populate_full_signature=True,
                    train=self.train.config if self.train else None,
                    evaluation=self.evaluation.config if self.evaluation else None,
                )
            )
        )


class Compose(RepresentationMixin):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input
