import pytest

from atria_core.types.base.data_model import BatchedBaseDataModel
from atria_core.types.data_instance.image import BatchedImageInstance, ImageInstance
from atria_core.types.generic.image import Image
from atria_core.types.generic.label import Label
from tests.types.factory import ImageInstanceFactory
from tests.types.tests_base import BaseDataModelTestBase


class TestImageInstance(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self) -> ImageInstance:
        return ImageInstanceFactory.build()

    def batched_model(self) -> type[BatchedBaseDataModel]:
        return BatchedImageInstance

    def tensor_fields(self) -> list[str]:
        return []

    def test_initialization(self, model_instance):
        # Check if the image attribute is an instance of Image
        image_instance = model_instance.to_tensor()

        assert isinstance(
            image_instance.image, Image
        ), "Image attribute is not an instance of Image"

        # Check if the label attribute is an instance of Label
        assert isinstance(
            image_instance.label, Label
        ), "Label attribute is not an instance of Label"

        assert image_instance.image.content.shape == (
            3,
            256,
            256,
        ), "Image content shape mismatch"

        assert image_instance.image.shape == (3, 256, 256), "Image size mismatch"
        assert image_instance.image.shape == (3, 256, 256), "Image size mismatch"
        assert image_instance.image.shape == (3, 256, 256), "Image size mismatch"
        assert image_instance.image.shape == (3, 256, 256), "Image size mismatch"
