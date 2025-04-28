from typing import List

import pytest
from data.structures.conftest import LabelFactory, LabelSequenceFactory
from data.structures.tests_base import BaseDataModelTestBase

from atria.data.structures.generic.label import (
    BatchedLabel,
    BatchedLabelSequence,
    LabelSequence,
)


class TestLabel(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self):
        return LabelFactory.build()

    def batched_model(self):
        return BatchedLabel

    def tensor_fields(self) -> List[str]:
        return ["value"]


class TestSequenceLabels(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self):
        return LabelSequenceFactory.build()

    def batched_model(self):
        return BatchedLabelSequence

    def tensor_fields(self) -> List[str]:
        return ["values"]


class TestSequenceFromLabels(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self):
        return LabelSequence.from_list(LabelFactory.batch(10))

    def batched_model(self):
        return BatchedLabelSequence

    def tensor_fields(self) -> List[str]:
        return ["values"]
