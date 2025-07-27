from abc import abstractmethod
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict

from atria_core.logger import get_logger
from atria_core.utilities.repr import RepresentationMixin

logger = get_logger(__name__)


class DataTransform(BaseModel, RepresentationMixin):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )
    _is_initialized: bool = False

    @cached_property
    def build_config(self) -> dict:
        return self._prepare_build_config()

    def initialize(self) -> None:
        """
        Initializes the data transform. This method should be overridden by subclasses
        to perform any necessary initialization.
        """
        if not self._is_initialized:
            self._lazy_post_init()
            self._is_initialized = True
        return self

    def _prepare_build_config(self):
        from hydra_zen import builds
        from omegaconf import OmegaConf

        cfg = builds(
            self.__class__,
            populate_full_signature=True,
            **{key: getattr(self, key) for key in self.model_fields_set},
        )
        return OmegaConf.to_container(OmegaConf.create(cfg))

    def _lazy_post_init(self) -> None:
        """
        Initializes the data transform. This method should be overridden by subclasses
        to perform any necessary initialization.
        """
        pass

    def __call__(
        self, input: Any | Mapping[str, Any] | list[Mapping[str, Any]]
    ) -> Any | Mapping[str, Any] | list[Mapping[str, Any]]:
        self.initialize()

        if isinstance(input, list):
            return [self._apply_transforms(s) for s in input]
        return self._apply_transforms(input)

    @abstractmethod
    def _apply_transforms(self, input: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _apply_transforms method."
        )


class ComposedTransform(DataTransform, RepresentationMixin):
    transforms: list[Callable]

    def _lazy_post_init(self) -> None:
        for tf in self.transforms:
            if hasattr(tf, "initialize"):
                tf.initialize()

    def _apply_transforms(self, input: Any) -> Any:
        for t in self.transforms:
            input = t(input)
        return input

    def _prepare_build_config(self):
        return {"transforms": [t.build_config for t in self.transforms]}


class DataTransformsDict(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="forbid"
    )

    train: DataTransform | dict[str, DataTransform] | ComposedTransform | None = None
    evaluation: DataTransform | dict[str, DataTransform] | ComposedTransform | None = (
        None
    )

    @property
    def composed_train(self) -> ComposedTransform | None:
        return self.compose("train") if self.train else None

    @property
    def composed_evaluation(self) -> ComposedTransform | None:
        return self.compose("evaluation") if self.evaluation else None

    def compose(self, key="train"):
        transform = getattr(self, key)
        if isinstance(transform, DataTransform):
            composed_transform = ComposedTransform(transforms=[transform])
        elif isinstance(transform, dict):
            composed_transform = ComposedTransform(transforms=list(transform.values()))
        else:
            composed_transform = transform  # Return as-is if not a known type

        return composed_transform.initialize()

    @property
    def build_config(self) -> dict:
        from hydra_zen import builds
        from omegaconf import OmegaConf

        return OmegaConf.to_container(
            OmegaConf.create(
                builds(
                    self.__class__,
                    populate_full_signature=True,
                    train=self.train.build_config if self.train else None,
                    evaluation=self.evaluation.build_config
                    if self.evaluation
                    else None,
                )
            )
        )
