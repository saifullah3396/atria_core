import types
from typing import Any, Generic, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
)
from rich.pretty import RichReprResult

from atria_core.logger.logger import get_logger
from atria_core.types.base._mixins._loadable import Loadable
from atria_core.types.base._mixins._raw_convertible import RawConvertible
from atria_core.types.base._mixins._repeatable import Repeatable
from atria_core.types.base._mixins._table_serializable import TableSerializable
from atria_core.types.base._mixins._tensor_convertible import TensorConvertible
from atria_core.types.base._mixins._to_device_convertible import ToDeviceConvertible
from atria_core.types.base.types import T_RawModel, T_TensorModel
from atria_core.utilities.repr import RepresentationMixin

logger = get_logger(__name__)


class BaseDataModel(RepresentationMixin, BaseModel):  # type: ignore[misc]
    pass


class RawDataModel(  # type: ignore[misc]
    BaseDataModel,
    TensorConvertible[T_TensorModel],
    Loadable,
    TableSerializable,
    Generic[T_TensorModel],
):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    @classmethod
    def _get_types(cls, field_annotation: Any) -> list[type]:
        """Extract non-None types from a field annotation."""
        origin = get_origin(field_annotation)
        args = get_args(field_annotation)
        if origin in {Union, types.UnionType} and len(args) > 1:
            return [arg for arg in args if arg is not types.NoneType]
        else:
            return [field_annotation]

    @classmethod
    def _verify_types(cls, type) -> None:
        non_none_types = cls._get_types(type)

        for t in non_none_types:
            if t == TensorDataModel:
                raise TypeError(
                    f"Field {type} is a subclass of {TensorDataModel}"
                    f"{RawDataModel} cannot contain {TensorDataModel} as children."
                )
            elif get_origin(t) in [list, tuple]:
                for tt in get_args(t):
                    if tt and issubclass(tt, TensorDataModel):
                        raise TypeError(
                            f"Field {type} is a list of {tt} or its children. "
                            f"{TensorDataModel} does not support nested lists of {TensorDataModel} as children."
                        )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        for _, field in cls.model_fields.items():
            cls._verify_types(field.annotation)

        cls.model_rebuild(force=True)

    def _set_skip_validation(self, name: str, value: Any) -> None:
        """Workaround to be able to set fields without validation."""
        attr = getattr(self.__class__, name, None)
        if isinstance(attr, property):
            attr.__set__(self, value)
        else:
            self.__dict__[name] = value
            self.__pydantic_fields_set__.add(name)

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, round_trip=True, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        return super().model_dump_json(*args, round_trip=True, **kwargs)


class TensorDataModel(  # type: ignore[misc]
    BaseDataModel,
    Repeatable,
    ToDeviceConvertible,
    RawConvertible[T_RawModel],
    RepresentationMixin,
    Generic[T_RawModel],
):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="ignore"
    )

    @classmethod
    def _get_types(cls, field_annotation: Any) -> list[type]:
        """Extract non-None types from a field annotation."""
        origin = get_origin(field_annotation)
        args = get_args(field_annotation)
        if origin in {Union, types.UnionType} and len(args) > 1:
            return [arg for arg in args if arg is not types.NoneType]
        else:
            return [field_annotation]

    @classmethod
    def _verify_types(cls, type) -> None:
        non_none_types = cls._get_types(type)

        for t in non_none_types:
            if t == RawDataModel:
                raise TypeError(
                    f"Field {type} is a subclass of {RawDataModel}"
                    f"{TensorDataModel} cannot contain {RawDataModel} as children."
                )
            elif get_origin(t) in [list, tuple]:
                for tt in get_args(t):
                    if tt and issubclass(tt, TensorDataModel | RawDataModel):
                        raise TypeError(
                            f"Field {type} is a list of {tt} or its children. "
                            f"{(cls)} does not support nested lists of {(TensorDataModel, RawDataModel)} as children."
                        )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        for _, field in cls.model_fields.items():
            cls._verify_types(field.annotation)

        cls.model_rebuild(force=True)

    @field_validator("*", mode="wrap")
    def ignore_validation(
        cls, value: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ):
        if info.context and "no_validation" in info.context:
            return value
        else:
            return handler(value)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the device to CPU after model creation."""
        import torch

        self._device = torch.device("cpu")

    def model_dump(self, *args, **kwargs):
        """
        Dumps the raw data model representation.

        Args:
            *args: Positional arguments for the dump method.
            **kwargs: Keyword arguments for the dump method.

        Returns:
            dict: The dumped representation.
        """
        return self.to_raw().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        """
        Dumps the raw data model representation as JSON.

        Args:
            *args: Positional arguments for the dump method.
            **kwargs: Keyword arguments for the dump method.

        Returns:
            str: The JSON representation.
        """
        return self.to_raw().model_dump_json(*args, **kwargs)

    def to_raw(self) -> T_RawModel:
        """
        Converts the current object and its fields to tensor representations.

        Returns:
            T_RawModel: An instance of the tensor model.
        """
        if self._is_batched:
            raise RuntimeError(
                "Cannot convert a batched TensorDataModel to raw. "
                "Please convert individual instances instead."
            )
        return super().to_raw()

    def __rich_repr__(self) -> RichReprResult:  # type: ignore[override]
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        import torch

        from atria_core.constants import _TORCH_PRINT_OPTIONS_PROFILE

        torch.set_printoptions(profile=_TORCH_PRINT_OPTIONS_PROFILE)
        for name, field_repr in self.__dict__.items():
            if name is None:
                yield field_repr
            else:
                yield name, field_repr
