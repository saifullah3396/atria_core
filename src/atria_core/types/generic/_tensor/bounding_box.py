from typing import TYPE_CHECKING, ClassVar, Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.generic._raw.bounding_box import BoundingBoxMode

if TYPE_CHECKING:
    from atria_core.types.generic._raw.bounding_box import BoundingBox, BoundingBoxList  # noqa


class TensorBoundingBoxBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: torch.Tensor
    mode: BoundingBoxMode = BoundingBoxMode.XYXY

    @property
    def is_valid(self) -> torch.Tensor:
        return (
            self.x1 >= 0
            and self.y1 >= 0
            and self.x2 > self.x1
            and self.y2 > self.y1
            and self.width > 0
            and self.height > 0
        )

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    @property
    def area(self) -> torch.Tensor:
        return self.width * self.height

    @property
    def x1(self) -> torch.Tensor:
        return self.value[(..., 0)]

    @x1.setter
    def x1(self, value: torch.Tensor | int | float | bool):
        self.value[(..., 0)] = value

    @property
    def y1(self) -> torch.Tensor:
        return self.value[(..., 1)]

    @y1.setter
    def y1(self, value: torch.Tensor | int | float | bool):
        self.value[(..., 1)] = value

    @property
    def x2(self) -> torch.Tensor:
        if self.mode == BoundingBoxMode.XYWH:
            return self.x1 + self.width
        else:
            return self.value[(..., 2)]

    @x2.setter
    def x2(self, value: torch.Tensor | int | float | bool):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            self.value[(..., 2)] = value

    @property
    def y2(self) -> torch.Tensor:
        if self.mode == BoundingBoxMode.XYWH:
            return self.y1 + self.height
        else:
            return self.value[(..., 3)]

    @y2.setter
    def y2(self, value: torch.Tensor | int | float | bool):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            self.value[(..., 3)] = value

    @property
    def width(self) -> torch.Tensor:
        if self.mode == BoundingBoxMode.XYWH:
            return self.value[(..., 2)]
        else:
            return self.x2 - self.x1

    @width.setter
    def width(self, value: torch.Tensor | int | float | bool):
        if self.mode == BoundingBoxMode.XYWH:
            self.value[(..., 2)] = value
        else:
            raise ValueError("Cannot set width directly in XYXY mode. Use x2 instead.")

    @property
    def height(self) -> torch.Tensor:
        if self.mode == BoundingBoxMode.XYWH:
            return self.value[(..., 3)]
        else:
            return self.y2 - self.y1

    @height.setter
    def height(self, value: torch.Tensor | int | float | bool):
        if self.mode == BoundingBoxMode.XYWH:
            self.value[(..., 3)] = value
        else:
            raise ValueError("Cannot set height directly in XYXY mode. Use y2 instead.")

    def normalize(self, width: torch.Tensor, height: torch.Tensor) -> Self:
        assert width > 0, "Width must be greater than 0."
        assert height > 0, "Height must be greater than 0."
        assert self.x1 <= width, "x1 must be less than or equal to width."
        assert self.y1 <= height, "y1 must be less than or equal to height."
        assert self.x2 <= width, "x2 must be less than or equal to width."
        assert self.y2 <= height, "y2 must be less than or equal to height."
        if self.mode == BoundingBoxMode.XYWH:
            self.x1 /= width
            self.y1 /= height
            self.width /= width
            self.height /= height
        else:
            self.x1 /= width
            self.y1 /= height
            self.x2 /= width
            self.y2 /= height
        return self


class TensorBoundingBox(TensorDataModel["BoundingBox"], TensorBoundingBoxBase):  # type: ignore[misc]
    _raw_model = "atria_core.types.generic._raw.bounding_box.BoundingBox"
    _batch_merge_fields: ClassVar[list[str] | None] = ["mode"]

    def switch_mode(self):
        assert not self._is_batched, "Cannot switch mode for batched bounding boxes."
        if self.mode == BoundingBoxMode.XYXY:
            self.value = torch.tensor([self.x1, self.y1, self.width, self.height])
            self.mode = BoundingBoxMode.XYWH
        else:
            self.value = torch.tensor([self.x1, self.y1, self.x2, self.y2])
            self.mode = BoundingBoxMode.XYXY
        return self

    @model_validator(mode="after")  # type: ignore[misc]
    def validate_value(self) -> Self:
        if self._is_batched:
            assert self.value.ndim == 2, (
                "Expected a 2D tensor for batched bounding boxes."
            )
            assert self.value.shape[-1] == 4, (
                "Batched bounding boxes must have dimension (N, 4)."
            )
        else:
            assert self.value.ndim == 1, (
                "Expected a 1D tensor of shape (4,) for bounding boxes."
            )
            assert self.value.shape[-1] == 4, "Bounding boxes must have dimension (4,)"
        return self


class TensorBoundingBoxList(TensorDataModel["BoundingBoxList"], TensorBoundingBoxBase):  # type: ignore[misc]
    _raw_model = "atria_core.types.generic._raw.bounding_box.BoundingBoxList"
    _batch_merge_fields: ClassVar[list[str] | None] = ["mode"]

    value: torch.Tensor
    mode: BoundingBoxMode = BoundingBoxMode.XYXY

    def switch_mode(self):
        assert not self._is_batched, "Cannot switch mode for batched bounding boxes."
        if self.mode == BoundingBoxMode.XYXY:
            self.value = torch.tensor([self.x1, self.y1, self.width, self.height])
            self.mode = BoundingBoxMode.XYWH
        else:
            self.value = torch.tensor([self.x1, self.y1, self.x2, self.y2])
            self.mode = BoundingBoxMode.XYXY
        return self

    @model_validator(mode="after")  # type: ignore[misc]
    def validate_value(self) -> Self:
        if self._is_batched:
            assert self.value.ndim == 3, (
                "Expected a 3D tensor for batched bounding boxes."
            )
            assert self.value.shape[-1] == 4, (
                "Batched bounding boxes must have dimension (B, L, 4)."
            )
        else:
            assert self.value.ndim == 2, (
                "Expected a 2D tensor of shape (L, 4) for bounding boxes."
            )
            assert self.value.shape[-1] == 4, (
                "Bounding boxes must have dimension (L, 4)"
            )
        return self
