import pytest
from data.structures.conftest import ImageInstanceFactory
from data.structures.tests_base import BaseDataModelTestBase

from atria.data.structures.base.data_model import BatchedBaseDataModel
from atria.data.structures.data_instance.image import (
    BatchedImageInstance,
    ImageInstance,
)
from atria.data.structures.generic.image import Image
from atria.data.structures.generic.label import Label


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
