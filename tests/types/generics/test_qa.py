from typing import List

import pytest

from atria_core.types.generic.question_answer_pair import (
    BatchedQuestionAnswerPair,
    BatchedQuestionAnswerPairSequence,
    QuestionAnswerPairSequence,
)
from tests.types.factory import (
    QuestionAnswerPairSequenceFactory,
    QuestionAnswerPairsFactory,
)
from tests.types.tests_base import BaseDataModelTestBase


class TestQuestionAnswerPair(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self):
        return QuestionAnswerPairsFactory.build()

    def batched_model(self):
        return BatchedQuestionAnswerPair

    def tensor_fields(self) -> List[str]:
        return []


class TestSequenceQuestionAnswerPairs(BaseDataModelTestBase):
    @pytest.fixture(params=[True, False])
    def from_qas(self, request):
        return request.param

    @pytest.fixture
    def model_instance(self, from_qas):
        if from_qas:
            return QuestionAnswerPairSequence.from_list(
                QuestionAnswerPairsFactory.batch(10)
            )
        else:
            return QuestionAnswerPairSequenceFactory.build()

    def batched_model(self):
        return BatchedQuestionAnswerPairSequence

    def tensor_fields(self) -> List[str]:
        return []
