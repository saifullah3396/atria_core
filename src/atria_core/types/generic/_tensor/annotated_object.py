from typing import TYPE_CHECKING, Any

import torch
from pydantic import field_validator

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.generic._tensor.bounding_box import TensorBoundingBox
from atria_core.types.generic._tensor.label import TensorLabel

if TYPE_CHECKING:
    from atria_core.types.generic._raw.annotated_object import AnnotatedObject  # noqa


class TensorAnnotatedObject(TensorDataModel["AnnotatedObject"]):
    _raw_model = "atria_core.types.generic._raw.annotated_object.AnnotatedObject"
    label: TensorLabel
    bbox: TensorBoundingBox
    segmentation: torch.Tensor | None = None
    iscrowd: bool = False

    @field_validator("segmentation", mode="before")
    @classmethod
    def validate_and_convert_segmentation(cls, value: Any) -> list[list[float]] | None:
        if isinstance(value, list):
            assert all(
                isinstance(poly, list) and len(poly) % 2 != 0 for poly in value
            ), (
                "Polygon segmentation must be of shape (N, 2M), where M is the number of points."
            )
        return value
