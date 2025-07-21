import enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Union

import pyarrow as pa
from pydantic import field_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import (
    ListFloatField,
    TableSchemaMetadata,
    _is_tensor_type,
    _tensor_validator,
)

if TYPE_CHECKING:
    import torch


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"  # (x1, y1, x2, y2)
    XYWH = "xywh"  # (x1, y1, width, height)


class BoundingBox(BaseDataModel):
    _batch_merge_fields: ClassVar[list[str]] = ["mode"]
    value: ListFloatField
    mode: Annotated[BoundingBoxMode, TableSchemaMetadata(pyarrow=pa.string())] = (
        BoundingBoxMode.XYXY
    )

    def switch_mode(self):
        assert not self._is_batched, "Cannot switch mode for batched bounding boxes."
        if _is_tensor_type(self.value):
            if self.mode == BoundingBoxMode.XYXY:
                self.value[..., 2], self.value[..., 3] = self.width, self.height
                self.mode = BoundingBoxMode.XYWH
            else:
                self.value[..., 2], self.value[..., 3] = self.x2, self.y2
                self.mode = BoundingBoxMode.XYXY
        else:
            if self.mode == BoundingBoxMode.XYXY:
                self.value = [self.x1, self.y1, self.width, self.height]
                self.mode = BoundingBoxMode.XYWH
            else:
                self.value = [self.x1, self.y1, self.x2, self.y2]
                self.mode = BoundingBoxMode.XYXY
        return self

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: Any) -> BoundingBoxMode:
        if isinstance(value, str):
            return BoundingBoxMode(value)
        return value

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> list[float]:
        assert len(value) == 4, "Expected a 1D list of shape (4,) for bounding boxes."
        return value

    @property
    def is_valid(self) -> Union[bool, "torch.Tensor"]:
        return (
            self.x1 >= 0
            and self.y1 >= 0
            and self.x2 > self.x1
            and self.y2 > self.y1
            and self.width > 0
            and self.height > 0
        )

    def shape(self) -> "torch.Size":
        if _is_tensor_type(self.value):
            return self.value.shape
        else:
            return (len(self.value),)

    @property
    def area(self) -> Union[float, "torch.Tensor"]:
        return self.width * self.height

    @property
    def x1(self) -> Union[float, "torch.Tensor"]:
        idx = (..., 0) if _is_tensor_type(self.value) else 0
        return self.value[idx]

    @x1.setter
    def x1(self, value: Union[float, "torch.Tensor"]):
        idx = (..., 0) if _is_tensor_type(self.value) else 0
        self.value[idx] = value

    @property
    def y1(self) -> Union[float, "torch.Tensor"]:
        idx = (..., 1) if _is_tensor_type(self.value) else 1
        return self.value[idx]

    @y1.setter
    def y1(self, value: Union[float, "torch.Tensor"]):
        idx = (..., 1) if _is_tensor_type(self.value) else 1
        self.value[idx] = value

    @property
    def x2(self) -> Union[float, "torch.Tensor"]:
        if self.mode == BoundingBoxMode.XYWH:
            return self.x1 + self.width
        else:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            return self.value[idx]

    @x2.setter
    def x2(self, value: Union[float, "torch.Tensor"]):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            self.value[idx] = value

    @property
    def y2(self) -> Union[float, "torch.Tensor"]:
        if self.mode == BoundingBoxMode.XYWH:
            return self.y1 + self.height
        else:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            return self.value[idx]

    @y2.setter
    def y2(self, value: Union[float, "torch.Tensor"]):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            self.value[idx] = value

    @property
    def width(self) -> Union[float, "torch.Tensor"]:
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            return self.value[idx]
        else:
            return self.x2 - self.x1

    @width.setter
    def width(self, value: Union[float, "torch.Tensor"]):
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            self.value[idx] = value
        else:
            raise ValueError("Cannot set width directly in XYXY mode. Use x2 instead.")

    @property
    def height(self) -> Union[float, "torch.Tensor"]:
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            return self.value[idx]
        else:
            return self.y2 - self.y1

    @height.setter
    def height(self, value: Union[float, "torch.Tensor"]):
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            self.value[idx] = value
        else:
            raise ValueError("Cannot set height directly in XYXY mode. Use y2 instead.")

    def normalize(self, width: float, height: float) -> "BoundingBox":
        """
        Normalizes the bounding box coordinates to the range [0, 1].

        Args:
            width (float): The width of the image or document.
            height (float): The height of the image or document.

        Returns:
            BoundingBox: The normalized bounding box.

        Raises:
            AssertionError: If the bounding box coordinates are invalid.
        """
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


class BoundingBoxList(BaseDataModel):
    value: Annotated[
        list[list[float]],
        _tensor_validator(2),
        TableSchemaMetadata(pyarrow=pa.list_(pa.list_(pa.float64()))),
    ]
    mode: Annotated[BoundingBoxMode, TableSchemaMetadata(pyarrow=pa.string())] = (
        BoundingBoxMode.XYXY
    )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: Any) -> BoundingBoxMode:
        if isinstance(value, str):
            return BoundingBoxMode(value)
        return value

    @classmethod
    def from_list(cls, bboxes: list[BoundingBox]) -> "BoundingBoxList":
        """
        Create a BoundingBoxList from a list of BoundingBox objects.
        """
        return cls(
            value=[bbox.value for bbox in bboxes],
            mode=bboxes[0].mode if bboxes else BoundingBoxMode.XYXY,
        )

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> list[float]:
        if isinstance(value, list):
            assert isinstance(value, list), "Expected a list of bounding boxes."
            if len(value) > 0:
                assert all(isinstance(bbox, list) for bbox in value), (
                    "Expected a 1D list of shape (4,) for bounding boxes."
                )
        elif _is_tensor_type(value):
            assert value.ndim == 2 and value.shape[1] == 4, (
                "Expected a 2D tensor with shape (N, 4) for bounding boxes."
            )
        return value
