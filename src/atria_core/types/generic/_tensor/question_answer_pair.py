from typing import TYPE_CHECKING

import torch
from pydantic import model_validator

from atria_core.types.base.data_model import TensorDataModel

if TYPE_CHECKING:
    from atria_core.types.generic._raw.question_answer_pair import (
        QuestionAnswerPair,  # noqa
        TokenizedQuestionAnswerPair,  # noqa
    )


class TensorQuestionAnswerPair(TensorDataModel["QuestionAnswerPair"]):
    _raw_model = "atria_core.types.generic._raw.question_answer_pair.QuestionAnswerPair"
    id: torch.Tensor
    question_text: str
    answer_start: torch.Tensor
    answer_end: torch.Tensor
    answer_text: list[str]

    @model_validator(mode="after")
    def validate_answer_field_lengths(self):
        assert len(self.answer_start) == len(self.answer_end), (
            "answer_start and answer_end must have the same length"
        )
        assert len(self.answer_start) == len(self.answer_text), (
            "answer_start and answer_text must have the same length"
        )
        return self


class TensorTokenizedQuestionAnswerPair(TensorDataModel["TokenizedQuestionAnswerPair"]):
    _raw_model = (
        "atria_core.types.generic._raw.question_answer_pair.TokenizedQuestionAnswerPair"
    )
    answer_starts: torch.Tensor
    answer_ends: torch.Tensor

    @model_validator(mode="after")
    def validate_answer_field_lengths(self):
        assert self.answer_starts.shape == self.answer_ends.shape, (
            "tokenized_answer_starts and tokenized_answer_ends must have the same shape"
        )
        return self
