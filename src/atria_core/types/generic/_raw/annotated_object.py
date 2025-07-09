from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
from pydantic import field_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.generic._raw.bounding_box import BoundingBox, BoundingBoxList
from atria_core.types.generic._raw.label import Label, LabelList
from atria_core.types.typing.common import BoolField, ListBoolField, TableSchemaMetadata

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.annotated_object import (
        TensorAnnotatedObject,  # noqa
        TensorAnnotatedObjectList,  # noqa
    )


def _validate_segmentation_field(v) -> list[list[float]] | None:
    if isinstance(v, dict) and "counts" in v and "size" in v:
        return v["counts"]
    elif isinstance(v, list):
        assert all(isinstance(poly, list) and len(poly) % 2 != 0 for poly in v), (
            "Polygon segmentation must be of shape (N, 2M), where M is the number of points."
        )
        return v
    return v


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
    iscrowd: BoolField

    @classmethod
    @field_validator("segmentation", mode="after")
    @classmethod
    def validate_and_convert_segmentation(cls, value: Any) -> list[list[float]] | None:
        if value is None:
            return None
        return _validate_segmentation_field(value)


class AnnotatedObjectList(RawDataModel["TensorAnnotatedObjectList"]):
    _tensor_model = (
        "atria_core.types.generic._tensor.annotated_object.TensorAnnotatedObjectList"
    )
    label: LabelList
    bbox: BoundingBoxList
    segmentation: Annotated[
        list[list[list[float]]] | None,
        TableSchemaMetadata(pyarrow=pa.list_(pa.list_(pa.float64()))),
    ] = None
    iscrowd: ListBoolField

    @classmethod
    @field_validator("segmentation", mode="after")
    @classmethod
    def validate_and_convert_segmentation(cls, value: Any) -> list[list[float]] | None:
        if value is None:
            return None

        assert isinstance(value, list), "Segmentation must be a list."
        for v in value:
            v = _validate_segmentation_field(v)
        return value

    @classmethod
    def from_list(cls, objects: list[AnnotatedObject]) -> "AnnotatedObjectList":
        """
        Create an AnnotatedObjectList from a list of AnnotatedObject objects.
        """
        segmentation = [obj.segmentation for obj in objects]
        if any(s is None for s in segmentation):
            segmentation = None
        return cls(
            label=LabelList.from_list([obj.label for obj in objects]),
            bbox=BoundingBoxList.from_list([obj.bbox for obj in objects]),
            segmentation=segmentation,
            iscrowd=[obj.iscrowd for obj in objects],
        )
