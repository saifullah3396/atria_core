from typing import TYPE_CHECKING, Self

import torch
from pydantic import model_validator

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.generic._tensor.bounding_box import (
    TensorBoundingBox,
    TensorBoundingBoxList,
)
from atria_core.types.generic._tensor.label import TensorLabel, TensorLabelList

if TYPE_CHECKING:
    from atria_core.types.generic._raw.annotated_object import (
        AnnotatedObject,  # noqa
        AnnotatedObjectList,  # noqa
    )  # noqa


class TensorAnnotatedObject(TensorDataModel["AnnotatedObject"]):
    _raw_model = "atria_core.types.generic._raw.annotated_object.AnnotatedObject"
    label: TensorLabel
    bbox: TensorBoundingBox
    segmentation: torch.Tensor | None = None
    iscrowd: torch.Tensor

    @model_validator(mode="after")
    def validate_and_convert_segmentation(self) -> Self:
        if self.segmentation is not None:
            assert self.segmentation.ndim == 2, (
                f"Expected a 2D tensor for segmentation, got {self.segmentation.ndim}D."
            )
            assert self.segmentation.shape[1] % 2 == 0, (
                "Expected segmentation to be of shape (N, 2M), where M is the number of points."
            )
        return self


class TensorAnnotatedObjectList(TensorDataModel["AnnotatedObjectList"]):
    _raw_model = "atria_core.types.generic._raw.annotated_object.AnnotatedObjectList"
    label: TensorLabelList
    bbox: TensorBoundingBoxList
    segmentation: torch.Tensor | None = None
    iscrowd: torch.Tensor

    @model_validator(mode="after")
    def validate_and_convert_segmentation(self) -> Self:
        if self.segmentation is not None:
            assert self.segmentation.ndim == 3, (
                f"Expected a 3D tensor for segmentation, got {self.segmentation.ndim}D."
            )
            assert self.segmentation.shape[1] % 2 == 0, (
                "Expected segmentation to be of shape (N, 2M), where M is the number of points. Got "
                f"shape {self.segmentation.shape}."
            )
            assert self.segmentation.shape[2] == 2, (
                "Expected segmentation points to be of shape (N, 2)."
            )
        return self
