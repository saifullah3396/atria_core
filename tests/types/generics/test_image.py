import copy
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from pydantic import ValidationError
from utilities import _compare_values

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.image import BatchedImage, Image
from tests.types.factory import ImageFactory
from tests.types.tests_base import BaseDataModelTestBase


class TestImage(BaseDataModelTestBase):
    @pytest.fixture(params=["pil", "pil_file", "numpy", "torch"])
    def backend(self, request):
        return request.param

    @pytest.fixture(params=[(100, 200), (150, 250), (64, 128)])
    def image_size(self, request):
        return request.param

    @pytest.fixture
    def model_instance(self, backend, image_size):
        ImageFactory.backend = backend
        ImageFactory.image_size = image_size
        return ImageFactory.build()

    @pytest.fixture
    def variable_size_model_instances(self, backend):
        instances = []
        for width, height in [(100, 200), (150, 250), (64, 128)]:
            ImageFactory.backend = backend
            ImageFactory.image_size = (width, height)
            instances.append(ImageFactory.build())
        return instances

    def batched_model(self) -> type[BaseDataModel]:
        return BatchedImage

    def tensor_fields(self) -> List[str]:
        return ["content"]

    def test_serialize(self, model_instance):
        instance = model_instance.to_tensor()
        seralized_instance = instance.model_dump()
        for name in self.tensor_fields():
            assert isinstance(
                seralized_instance[name], (bytes, str)
            ), f"Field {name} is not a numpy array: {seralized_instance[name]}"
        assert isinstance(
            seralized_instance, dict
        ), "Serialized instance is not a dictionary"

    def test_invalid_image(self):
        with pytest.raises(
            FileNotFoundError,
        ):
            Image(
                file_path="test.png",
                content=None,
            ).to_tensor()

    def test_image_types(self):
        with pytest.raises(ValidationError):
            torch_image = Image(
                content=torch.randn((3)),
            ).to_tensor()

        torch_image = Image(
            content=torch.randn((3, 200, 100)),
        ).to_tensor()
        pil_image = Image(
            content=PILImage.new("RGB", (100, 200), color="white"),
        ).to_tensor()
        np_image = Image(
            content=np.random.randn(200, 100, 3),
        ).to_tensor()

        for image in [torch_image, pil_image, np_image]:
            assert image.shape == (3, 200, 100)
            assert image.size == (100, 200)
            assert image.width == 100
            assert image.height == 200

    def test_batched_variable_sizes(self, variable_size_model_instances):
        with pytest.raises(AssertionError):
            variable_size_model_instances[0].batched(variable_size_model_instances)
        instances = [
            x.to_tensor() for x in copy.deepcopy(variable_size_model_instances)
        ]
        batched_instances = instances[0].batched(instances)
        assert isinstance(batched_instances, BaseDataModel)
        assert isinstance(
            batched_instances,
            self.batched_model(),
        )

        def _validate_batched_values(batched_instances, instances):
            for attr_name, batched_value in batched_instances.__dict__.items():
                if attr_name == "batch_size":
                    continue
                if isinstance(batched_value, torch.Tensor):
                    continue
                if isinstance(batched_value, list) and isinstance(
                    batched_value[0], torch.Tensor
                ):
                    continue
                if isinstance(batched_value, BaseDataModel):
                    child_instances = [
                        getattr(instance, attr_name) for instance in instances
                    ]
                    _validate_batched_values(batched_value, child_instances)
                else:
                    if batched_value is None:
                        continue
                    assert isinstance(
                        batched_value, list
                    ), f"Field {attr_name} is not a list"
                    assert len(batched_value) == len(
                        instances
                    ), f"Field {attr_name} has different lengths"
                    for i in range(len(batched_value)):
                        if isinstance(batched_value[i], list):
                            for j in range(len(batched_value[i])):
                                if isinstance(instances[i], list):
                                    _compare_values(
                                        batched_value[i][j],
                                        getattr(instances[i][j], attr_name),
                                    )
                                else:
                                    _compare_values(
                                        batched_value[i][j],
                                        getattr(instances[i], attr_name)[j],
                                    )
                        else:
                            _compare_values(
                                batched_value[i], getattr(instances[i], attr_name)
                            )

        for name in self.tensor_fields():
            value = getattr(batched_instances, name)
            assert isinstance(value, list)
            assert len(value) == len(instances)
            for i in range(1, len(instances)):
                _compare_values(value[i], getattr(instances[i], name))
        _validate_batched_values(batched_instances, instances)
