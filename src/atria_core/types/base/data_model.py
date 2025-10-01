import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict

from atria_core.logger.logger import get_logger
from atria_core.types.base._mixins._file_path_convertible import FilePathConvertible
from atria_core.types.base._mixins._loadable import Loadable
from atria_core.types.base._mixins._repeatable import Repeatable
from atria_core.types.base._mixins._table_serializable import TableSerializable
from atria_core.types.base._mixins._tensor_convertible import TensorConvertible
from atria_core.types.base._mixins._to_device_convertible import ToDeviceConvertible
from atria_core.utilities.repr import RepresentationMixin

logger = get_logger(__name__)


class PydanticBase(RepresentationMixin, BaseModel):  # type: ignore[misc]
    pass


class BaseDataModel(  # type: ignore[misc]
    PydanticBase,
    TensorConvertible,
    Loadable,
    Repeatable,
    ToDeviceConvertible,
    TableSerializable,
    FilePathConvertible,
):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        strict=True,
    )

    def _set_skip_validation(self, name: str, value: Any) -> None:
        """Workaround to be able to set fields without validation."""
        attr = getattr(self.__class__, name, None)
        if isinstance(attr, property):
            attr.__set__(self, value)
        else:
            self.__dict__[name] = value
            self.__pydantic_fields_set__.add(name)

    @classmethod
    def _get_types(cls, field_annotation: Any) -> list[type]:
        """Extract non-None types from a field annotation."""
        origin = get_origin(field_annotation)
        args = get_args(field_annotation)
        if origin in {Union, types.UnionType} and len(args) > 1:
            return [arg for arg in args if arg is not types.NoneType]
        else:
            return [field_annotation]

    def model_dump(self, *args, **kwargs):
        self.to_raw()
        return super().model_dump(*args, round_trip=True, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        self.to_raw()
        return super().model_dump_json(*args, round_trip=True, **kwargs)
