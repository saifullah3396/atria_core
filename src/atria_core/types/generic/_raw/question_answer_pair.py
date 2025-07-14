from typing import TYPE_CHECKING

from pydantic import model_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.typing.common import (
    IntField,
    ListIntField,
    ListStrField,
    StrField,
)

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.question_answer_pair import (
        TensorQuestionAnswerPair,  # noqa
        TensorTokenizedQuestionAnswerPair,  # noqa
    )


class QuestionAnswerPair(RawDataModel):
    id: IntField
    question_text: StrField
    answer_start: ListIntField
    answer_end: ListIntField
    answer_text: ListStrField

    @model_validator(mode="after")
    def validate_answer_field_lengths(self):
        assert len(self.answer_start) == len(self.answer_end), (
            "answer_start and answer_end must have the same length"
        )
        assert len(self.answer_start) == len(self.answer_text), (
            "answer_start and answer_text must have the same length"
        )
        return self


class TokenizedQuestionAnswerPair(RawDataModel):
    answer_starts: ListIntField
    answer_ends: ListIntField

    @model_validator(mode="after")
    def validate_answer_field_lengths(self):
        assert len(self.answer_starts) == len(self.answer_ends), (
            "tokenized_answer_starts and tokenized_answer_ends must have the same length"
        )
        return self
