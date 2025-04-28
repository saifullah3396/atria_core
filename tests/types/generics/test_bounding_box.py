from typing import List

import pytest
from data.structures.conftest import BoundingBoxFactory, SequenceBoundingBoxesFactory
from data.structures.tests_base import BaseDataModelTestBase

from atria.data.structures.base.data_model import BaseDataModel
from atria.data.structures.generic.bounding_box import (
    BatchedBoundingBox,
    BatchedBoundingBoxSequence,
    BoundingBoxMode,
    BoundingBoxSequence,
)


class TestBoundingBox(BaseDataModelTestBase):
    @pytest.fixture(params=["np", "list", "torch"])
    def backend(self, request):
        return request.param

    @pytest.fixture
    def model_instance(self, backend):
        BoundingBoxFactory.backend = backend
        return BoundingBoxFactory.build()

    def batched_model(self) -> type[BaseDataModel]:
        return BatchedBoundingBox

    def tensor_fields(self) -> List[str]:
        return ["value"]


class TestSequenceBoundingBox(BaseDataModelTestBase):
    @pytest.fixture(params=["np", "list", "torch"])
    def backend(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def from_boxes(self, request):
        return request.param

    @pytest.fixture
    def model_instance(self, backend, from_boxes):
        if from_boxes:
            BoundingBoxFactory.backend = backend
            return BoundingBoxSequence.from_list(
                BoundingBoxFactory.batch(10, mode=BoundingBoxMode.XYXY)
            )
        else:
            SequenceBoundingBoxesFactory.backend = backend
            return SequenceBoundingBoxesFactory.build()

    def batched_model(self) -> type[BaseDataModel]:
        return BatchedBoundingBoxSequence

    def tensor_fields(self) -> List[str]:
        return ["value"]
