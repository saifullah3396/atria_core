from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
from pydantic import field_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.generic._raw.bounding_box import BoundingBox
from atria_core.types.generic._raw.label import Label
from atria_core.types.typing.common import BoolField, TableSchemaMetadata

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.annotated_object import (
        TensorAnnotatedObject,  # noqa
    )


class AnnotatedObject(RawDataModel["TensorAnnotatedObject"]):
    _tensor_model = (
        "atria_core.types.generic._tensor.annotated_object.TensorAnnotatedObject"
    )
    label: Label
    bbox: BoundingBox
    segmentation: Annotated[
        list[list[float]] | None,
        TableSchemaMetadata(pyarrow=pa.list_(pa.list_(pa.float64()))),
    ] = None
    iscrowd: BoolField = False

    @classmethod
    @field_validator("segmentation", mode="before")
    @classmethod
    def validate_and_convert_segmentation(cls, value: Any) -> list[list[float]] | None:
        if isinstance(value, dict) and "counts" in value and "size" in value:
            return value["counts"]
        elif isinstance(value, list):
            assert all(
                isinstance(poly, list) and len(poly) % 2 != 0 for poly in value
            ), (
                "Polygon segmentation must be of shape (N, 2M), where M is the number of points."
            )
            return value
        return value
