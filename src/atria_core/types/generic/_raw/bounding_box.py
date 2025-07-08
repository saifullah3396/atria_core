import enum
from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
from pydantic import field_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.typing.common import ListFloatField, TableSchemaMetadata

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.bounding_box import TensorBoundingBox  # noqa


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"  # (x1, y1, x2, y2)
    XYWH = "xywh"  # (x1, y1, width, height)


class BoundingBox(RawDataModel["TensorBoundingBox"]):
    _tensor_model = "atria_core.types.generic._tensor.bounding_box.TensorBoundingBox"
    value: ListFloatField
    mode: Annotated[BoundingBoxMode, TableSchemaMetadata(pyarrow=pa.string())] = (
        BoundingBoxMode.XYXY
    )

    def switch_mode(self):
        if self.mode == BoundingBoxMode.XYXY:
            self.value = [self.x1, self.y1, self.width, self.height]
            self.mode = BoundingBoxMode.XYWH
        else:
            self.value = [self.x1, self.y1, self.x2, self.y2]
            self.mode = BoundingBoxMode.XYXY
        return self

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> list[float]:
        assert len(value) == 4, "Expected a 1D list of shape (4,) for bounding boxes."
        return value

    @property
    def is_valid(self) -> bool:
        return (
            self.x1 >= 0
            and self.y1 >= 0
            and self.x2 > self.x1
            and self.y2 > self.y1
            and self.width > 0
            and self.height > 0
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def x1(self) -> float:
        return self.value[0]

    @x1.setter
    def x1(self, value: float):
        self.value[0] = value

    @property
    def y1(self) -> float:
        return self.value[1]

    @y1.setter
    def y1(self, value: float):
        self.value[1] = value

    @property
    def x2(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.x1 + self.width
        else:
            return self.value[2]

    @x2.setter
    def x2(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            self.value[2] = value

    @property
    def y2(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.y1 + self.height
        else:
            return self.value[3]

    @y2.setter
    def y2(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            self.value[3] = value

    @property
    def width(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.value[2]
        else:
            return self.x2 - self.x1

    @width.setter
    def width(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            self.value[2] = value
        else:
            raise ValueError("Cannot set width directly in XYXY mode. Use x2 instead.")

    @property
    def height(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.value[3]
        else:
            return self.y2 - self.y1

    @height.setter
    def height(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            self.value[3] = value
        else:
            raise ValueError("Cannot set height directly in XYXY mode. Use y2 instead.")

    def normalize(self, width: float, height: float) -> "BoundingBox":
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
