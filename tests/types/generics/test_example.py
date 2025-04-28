from typing import List, Type

import pytest
from data.structures.tests_base import (
    BaseDataModelTestBase,
    BatchedMockDataModelParent,
    MockDataModelChild,
    MockDataModelParent,
)

from atria.data.structures.base.data_model import BaseDataModel


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
