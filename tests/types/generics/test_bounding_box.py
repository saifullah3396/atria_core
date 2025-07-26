import pyarrow as pa
import pytest

from atria_core.types.factory import BoundingBoxFactory
from atria_core.types.generic.bounding_box import BoundingBox, BoundingBoxMode
from tests.types.data_model_test_base import DataModelTestBase
from tests.utilities.common import _assert_values_equal


class TestBoundingBox(DataModelTestBase):
    """
    Test class for BoundingBox.
    """

    factory = BoundingBoxFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"value": pa.list_(pa.float64()), "mode": pa.string()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"value": pa.list_(pa.float64()), "mode": pa.string()}


#########################################################
# Basic BoundingBox Tests
#########################################################
@pytest.fixture
def valid_bbox() -> BoundingBox:
    return BoundingBox(
        value=[10.0, 20.0, 30.0, 40.0], mode=BoundingBoxMode.XYXY
    )  # ignore[arg-type]


def test_initialization(valid_bbox: BoundingBox) -> None:
    assert valid_bbox.value == [10.0, 20.0, 30.0, 40.0]
    assert valid_bbox.mode == BoundingBoxMode.XYXY


def test_bbox_is_valid(valid_bbox: BoundingBox) -> None:
    assert valid_bbox.is_valid is True

    invalid_bbox = BoundingBox(
        value=[-10.0, -20.0, -30.0, -40.0], mode=BoundingBoxMode.XYXY
    )  # ignore[arg-type]
    assert invalid_bbox.is_valid is False

    invalid_bbox = BoundingBox(
        value=[10.0, 20.0, 5.0, 40.0], mode=BoundingBoxMode.XYXY
    )  # ignore[arg-type]
    assert invalid_bbox.is_valid is False


def test_tensor_bbox(valid_bbox: BoundingBox) -> None:
    import torch

    tensor_bbox = valid_bbox.to_tensor()
    assert tensor_bbox.value.shape == (4,)
    target = torch.tensor([10.0, 20.0, 30.0, 40.0])
    _assert_values_equal(tensor_bbox.value, target)


def test_tensor_batched_bboxes(valid_bbox: BoundingBox) -> None:
    tensor_bbox = valid_bbox.to_tensor()
    tensor_bbox_batched = tensor_bbox.batched([tensor_bbox, tensor_bbox, tensor_bbox])
    assert tensor_bbox_batched.value.shape == (3, 4)
    assert tensor_bbox_batched._is_batched is True
    assert tensor_bbox_batched.mode == BoundingBoxMode.XYXY
    assert tensor_bbox_batched.x1.tolist() == [10.0, 10.0, 10.0]
    assert tensor_bbox_batched.y1.tolist() == [20.0, 20.0, 20.0]
    assert tensor_bbox_batched.x2.tolist() == [30.0, 30.0, 30.0]
    assert tensor_bbox_batched.y2.tolist() == [40.0, 40.0, 40.0]
    assert tensor_bbox_batched.width.tolist() == [20.0, 20.0, 20.0]
    assert tensor_bbox_batched.height.tolist() == [20.0, 20.0, 20.0]

    valid_bbox.switch_mode()

    assert tensor_bbox_batched.x1.tolist() == [10.0, 10.0, 10.0]
    assert tensor_bbox_batched.y1.tolist() == [20.0, 20.0, 20.0]
    assert tensor_bbox_batched.x2.tolist() == [30.0, 30.0, 30.0]
    assert tensor_bbox_batched.y2.tolist() == [40.0, 40.0, 40.0]
    assert tensor_bbox_batched.width.tolist() == [20.0, 20.0, 20.0]
    assert tensor_bbox_batched.height.tolist() == [20.0, 20.0, 20.0]


def test_tensor_batched_bboxes_manipulated(valid_bbox: BoundingBox) -> None:
    tensor_bbox = valid_bbox.to_tensor()
    tensor_bbox_batched = tensor_bbox.batched([tensor_bbox, tensor_bbox, tensor_bbox])
    assert tensor_bbox_batched.value.shape == (3, 4)
    assert tensor_bbox_batched._is_batched is True
    assert tensor_bbox_batched.mode == BoundingBoxMode.XYXY

    assert tensor_bbox_batched.x1.tolist() == [0, 0, 0]
    assert tensor_bbox_batched.y1.tolist() == [10.0, 20.0, 30.0]
    assert tensor_bbox_batched.x2.tolist() == [30.0, 30.0, 30.0]
    assert tensor_bbox_batched.y2.tolist() == [40.0, 40.0, 40.0]
    assert tensor_bbox_batched.width.tolist() == [30.0, 30.0, 30.0]
    assert tensor_bbox_batched.height.tolist() == [30.0, 20.0, 10.0]

    valid_bbox = valid_bbox.switch_mode()

    assert tensor_bbox_batched.x1.tolist() == [0, 0, 0]
    assert tensor_bbox_batched.y1.tolist() == [10.0, 20.0, 30.0]
    assert tensor_bbox_batched.x2.tolist() == [30.0, 30.0, 30.0]
    assert tensor_bbox_batched.y2.tolist() == [40.0, 40.0, 40.0]
    assert tensor_bbox_batched.width.tolist() == [30.0, 30.0, 30.0]
    assert tensor_bbox_batched.height.tolist() == [30.0, 20.0, 10.0]
