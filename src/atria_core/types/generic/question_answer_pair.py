"""
Question-Answer Module

This module defines the `QuestionAnswerPair`, `BatchedQuestionAnswerPair`, `SequenceQuestionAnswerPairs`, and `BatchedSequenceQuestionAnswerPairs` classes, which represent question-answer pairs in a dataset. These classes include fields for the question text, answer text, start and end positions, and correctness. The batched counterparts handle multiple instances of question-answer pairs.

Classes:
    - QuestionAnswerPair: Represents a single question-answer pair.
    - BatchedQuestionAnswerPair: Represents a batch of question-answer pairs.
    - SequenceQuestionAnswerPairs: Represents a sequence of question-answer pairs.
    - BatchedSequenceQuestionAnswerPairs: Represents a batch of sequences of question-answer pairs.

Dependencies:
    - typing: For type annotations.
    - atria_core.data_types.base.data_model: For the base data model class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""



from atria_core.types.base.data_model import BaseDataModel


class QuestionAnswerPair(BaseDataModel):
    """
    A class for representing a single question and its corresponding answer.

    Attributes:
        id (int): The unique identifier for the question-answer pair.
        question_text (str): The text of the question.
        answer_start (int): The start position of the answer in the text.
        answer_end (int): The end position of the answer in the text.
        answer_text (str): The text of the answer.
        answer_is_correct (bool): Indicates whether the answer is correct.
    """

    id: int
    question_text: str
    answer_start: int
    answer_end: int
    answer_text: str
    answer_is_correct: bool
