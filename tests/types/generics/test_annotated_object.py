from typing import List

import pytest
from data.structures.conftest import (
    AnnotatedObjectFactory,
    AnnotatedObjectSequenceFactory,
)
from data.structures.tests_base import BaseDataModelTestBase

from atria.data.structures.base.data_model import BaseDataModel
from atria.data.structures.generic.annotated_object import (
    AnnotatedObjectSequence,
    BatchedAnnotatedObject,
    BatchedAnnotatedObjectSequence,
)


class TestAnnotatedObject(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self):
        return AnnotatedObjectFactory.build()

    def batched_model(self) -> type[BaseDataModel]:
        return BatchedAnnotatedObject

    def tensor_fields(self) -> List[str]:
        return ["segmentation"]


class TestSequenceAnnotatedObjects(BaseDataModelTestBase):
    @pytest.fixture(params=[True, False])
    def from_annotated_objects(self, request):
        return request.param

    def batched_model(self) -> type[BaseDataModel]:
        return BatchedAnnotatedObjectSequence

    def tensor_fields(self) -> List[str]:
        return ["segmentation"]

    @pytest.fixture
    def model_instance(self, from_annotated_objects):
        if from_annotated_objects:
            return AnnotatedObjectSequence.from_list(AnnotatedObjectFactory.batch(10))
        else:
            return AnnotatedObjectSequenceFactory.build()
