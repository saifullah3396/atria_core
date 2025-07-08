from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
import torch  # noqa
from PIL import Image as PILImageModule

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.factory import ImageFactory
from atria_core.types.generic._raw.image import Image
from atria_core.types.generic._tensor.image import TensorImage
from atria_core.utilities.encoding import _image_to_bytes
from tests.types.data_model_test_base import DataModelTestBase
from tests.utilities.common import _assert_values_equal


class TestImage(DataModelTestBase):
    """
    Test class for Image.
    """

    factory = ImageFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "file_path": pa.string(),
            "content": pa.binary(),
            "width": pa.int64(),
            "height": pa.int64(),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "file_path": pa.string(),
            "content": pa.binary(),
            "width": pa.int64(),
            "height": pa.int64(),
        }

    def test_to_from_tensor(self, model_instance: RawDataModel) -> None:
        """
        Test the conversion of the model instance to a tensor.
        """
        model_instance.load()
        tensor_model = model_instance.to_tensor()
        assert tensor_model is not None, "Tensor conversion returned None"
        assert isinstance(tensor_model, model_instance.tensor_data_model()), (
            "Tensor conversion did not return the expected tensor model type"
        )
        roundtrip_model = tensor_model.to_raw()
        assert isinstance(roundtrip_model, model_instance.__class__), (
            "Raw conversion did not return a RawDataModel"
        )

        set_fields = roundtrip_model.model_fields_set
        roundtrip_model = roundtrip_model.model_dump(include=set_fields)
        original_data = model_instance.model_dump(include=set_fields)
        _assert_values_equal(roundtrip_model, original_data)


#########################################################
# Basic Image Tests
#########################################################
@pytest.fixture
def valid_image_path(tmp_path: Path) -> str:
    image_path = tmp_path / "test_image.jpg"
    PILImageModule.new("RGB", (100, 100)).save(image_path)
    return str(image_path)


@pytest.fixture
def valid_raw_image(valid_image_path: str) -> Image:
    return Image(file_path=valid_image_path)


def test_image_initialization(valid_image_path: str) -> None:
    raw_image = Image(file_path=valid_image_path)
    assert str(raw_image.file_path) == valid_image_path
    assert raw_image.content is None


def test_load_from_file(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    assert valid_raw_image.content is not None
    assert valid_raw_image.size == (100, 100)


@patch("requests.get")
def test_load_from_url(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = _image_to_bytes(PILImageModule.new("RGB", (100, 100)))
    mock_get.return_value = mock_response

    raw_image = Image(file_path="https://example.com/test_image.jpg")
    raw_image.load()
    assert raw_image.content is not None


def test_unload(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    valid_raw_image.unload()
    assert valid_raw_image.content is None


def test_to_rgb(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    valid_raw_image.content = valid_raw_image.content.convert("L")
    assert valid_raw_image.channels == 1
    valid_raw_image.to_rgb()
    assert valid_raw_image.channels == 3


def test_resize(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    valid_raw_image.resize(50, 50)
    assert valid_raw_image.size == (50, 50)


def test_shape(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    assert valid_raw_image.shape == (3, 100, 100)


def test_size(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    assert valid_raw_image.size == (100, 100)


def test_channels(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    assert valid_raw_image.channels == 3


def test_to_tensor(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    tensor = valid_raw_image.to_tensor()
    assert tensor is not None
    assert tensor.shape == (3, 100, 100)


def test_to_tensor_without_content() -> None:
    raw_image = Image(file_path="dummy_path.jpg")
    with pytest.raises(RuntimeError):
        raw_image.load()
        raw_image.tensor_data_model()


def test_to_tensor_grayscale(valid_raw_image: Image) -> None:
    valid_raw_image.load()
    valid_raw_image.content = valid_raw_image.content.convert("L")
    tensor = valid_raw_image.to_tensor()
    assert tensor.shape == (1, 100, 100)


def test_to_tensor_rgba(tmp_path: Path) -> None:
    image_path = tmp_path / "test_rgba.png"
    PILImageModule.new("RGBA", (100, 100)).save(image_path)
    raw_image = Image(file_path=str(image_path))
    raw_image.load()
    tensor = raw_image.to_tensor()
    assert tensor.shape == (4, 100, 100)


#########################################################
# TensorImage Tests
#########################################################
@pytest.fixture
def valid_rgb_tensor_image() -> TensorImage:
    return TensorImage(content=PILImageModule.new("RGB", (32, 32)))


@pytest.fixture
def valid_gray_tensor_image() -> TensorImage:
    return TensorImage(content=PILImageModule.new("L", (32, 32)))


def test_tensor_image_initialization_tensor() -> None:
    tensor_image = TensorImage(content=torch.randn(3, 32, 32))
    assert tensor_image.shape == (3, 32, 32)


def test_tensor_image_initialization_pil() -> None:
    tensor_image = TensorImage(content=PILImageModule.new("RGB", (32, 32)))
    assert tensor_image.shape == (3, 32, 32)


def test_tensor_image_initialization_np() -> None:
    import numpy as np

    tensor_image = TensorImage(content=np.random.randn(32, 32, 3))
    assert tensor_image.shape == (3, 32, 32)


def test_tensor_image_to_raw(valid_rgb_tensor_image: TensorImage) -> None:
    raw_image = valid_rgb_tensor_image.to_raw()
    assert raw_image.shape == (3, 32, 32)


def test_tensor_image_resize(valid_rgb_tensor_image: TensorImage) -> None:
    resized = valid_rgb_tensor_image.resize(4, 4)
    assert resized.shape == (3, 4, 4)


def test_batched_images(valid_rgb_tensor_image: TensorImage) -> None:
    batched_image = valid_rgb_tensor_image.batched(
        [valid_rgb_tensor_image, valid_rgb_tensor_image, valid_rgb_tensor_image]
    )
    assert batched_image is not None
    assert batched_image._is_batched is True
    assert len(batched_image.content) == 3
    assert batched_image.content.shape == (3, 3, 32, 32)


def test_tensor_image_to_rgb(valid_gray_tensor_image: TensorImage) -> None:
    valid_rgb_tensor_image = valid_gray_tensor_image.to_rgb()
    assert valid_rgb_tensor_image.channels == 3
    assert valid_rgb_tensor_image.content.shape[0] == 3  # Channels dimension
    assert valid_rgb_tensor_image.content.shape[1:] == (32, 32)


def test_batched_to_rgb(valid_gray_tensor_image: TensorImage) -> None:
    batch_gray_image = valid_gray_tensor_image.batched(
        [valid_gray_tensor_image, valid_gray_tensor_image, valid_gray_tensor_image]
    )
    batch_rgb_image = batch_gray_image.to_rgb()
    assert batch_rgb_image.channels == 3
    assert batch_rgb_image.content.shape[1] == 3  # Channels dimension
    assert batch_rgb_image.content.shape[2:] == (32, 32)  # Height and Width dimensions
