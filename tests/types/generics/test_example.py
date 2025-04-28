from typing import List, Type

import pytest

from atria_core.types.base.data_model import BaseDataModel
from tests.types.tests_base import (
    BaseDataModelTestBase,
    BatchedMockDataModelParent,
    MockDataModelChild,
    MockDataModelParent,
)


class TestMockDataModelParent(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self):
        return MockDataModelParent(
            required_integer_attribute=1,
            required_integer_list_attribute=[1, 2, 3],
            example_data_model_child=MockDataModelChild(
                required_integer_attribute=10,
                required_integer_list_attribute=[10, 20, 30],
            ),
        )

    def batched_model(self) -> Type[BaseDataModel]:
        return BatchedMockDataModelParent

    def tensor_fields(self) -> List[str]:
        return [
            "attribute1",
            "attribute2",
            "tensor_attribute",
            "tensor_list_attribute",
        ]
