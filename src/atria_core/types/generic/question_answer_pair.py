from pydantic import model_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import (
    IntField,
    ListIntField,
    ListStrField,
    StrField,
)


class QuestionAnswerPair(BaseDataModel):
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
