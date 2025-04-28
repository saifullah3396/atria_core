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

from typing import List

from pydantic import model_validator

from atria_core.types.base.data_model import BaseDataModel, BatchedBaseDataModel


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

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedQuestionAnswerPair":
        """
        Constructs a batch of QuestionAnswerPair objects from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments containing lists of attributes.

        Returns:
            BatchedQuestionAnswerPair: A BatchedQuestionAnswerPair object containing lists of attributes.
        """
        return BatchedQuestionAnswerPair(**kwargs)


class BatchedQuestionAnswerPair(BatchedBaseDataModel):
    """
    A class for representing a batch of question-answer pairs.

    Attributes:
        id (List[int]): The unique identifiers for the question-answer pairs.
        question_text (List[str]): The list of question texts.
        answer_start (List[int]): The start positions of the answers in the text.
        answer_end (List[int]): The end positions of the answers in the text.
        answer_text (List[str]): The texts of the answers.
        answer_is_correct (List[bool]): Indicates whether each answer is correct.
    """

    id: List[int]
    question_text: List[str]
    answer_start: List[int]
    answer_end: List[int]
    answer_text: List[str]
    answer_is_correct: List[bool]


class QuestionAnswerPairSequence(BaseDataModel):
    """
    A class for representing a sequence of question-answer pairs.

    Attributes:
        id (List[int]): The unique identifiers for the question-answer pairs in the sequence.
        question_text (List[str]): The list of question texts in the sequence.
        answer_start (List[int]): The start positions of the answers in the sequence.
        answer_end (List[int]): The end positions of the answers in the sequence.
        answer_text (List[str]): The texts of the answers in the sequence.
        answer_is_correct (List[bool]): Indicates whether each answer in the sequence is correct.
    """

    id: List[int]
    question_text: List[str]
    answer_start: List[int]
    answer_end: List[int]
    answer_text: List[str]
    answer_is_correct: List[bool]

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedQuestionAnswerPairSequence":
        """
        Constructs a batch of SequenceQuestionAnswerPairs objects from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments containing lists of attributes.

        Returns:
            BatchedSequenceQuestionAnswerPairs: A BatchedSequenceQuestionAnswerPairs object containing lists of attributes.
        """
        return BatchedQuestionAnswerPairSequence(**kwargs)

    @classmethod
    def from_list(cls, value: List[QuestionAnswerPair]) -> "QuestionAnswerPairSequence":
        """
        Constructs a new instance of SequenceQuestionAnswerPairs from a list of QuestionAnswerPair objects.

        Args:
            qa_pairs (List[QuestionAnswerPair]): A list of QuestionAnswerPair instances.

        Returns:
            SequenceQuestionAnswerPairs: A new instance of SequenceQuestionAnswerPairs constructed from the list of question-answer pairs.
        """
        assert len(value) > 0, "The list of question-answer pairs cannot be empty."
        if isinstance(value, list) and all(
            isinstance(x, QuestionAnswerPair) for x in value
        ):
            return cls(
                id=[x.id for x in value],
                question_text=[x.question_text for x in value],
                answer_start=[x.answer_start for x in value],
                answer_end=[x.answer_end for x in value],
                answer_text=[x.answer_text for x in value],
                answer_is_correct=[x.answer_is_correct for x in value],
            )
        raise TypeError(
            f"Expected a list of QuestionAnswerPair instances, got {type(value)}"
        )

    def to_list(self) -> List[QuestionAnswerPair]:
        """
        Converts the SequenceQuestionAnswerPairs instance to a list of QuestionAnswerPair objects.

        Returns:
            List[QuestionAnswerPair]: A list of QuestionAnswerPair instances.
        """
        return [
            QuestionAnswerPair(
                id=self.id[i],
                question_text=self.question_text[i],
                answer_start=self.answer_start[i],
                answer_end=self.answer_end[i],
                answer_text=self.answer_text[i],
                answer_is_correct=self.answer_is_correct[i],
            )
            for i in range(len(self.id))
        ]

    @model_validator(mode="after")
    def validate_input_sizes(self):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, list) and len(attr_value) != len(self.id):
                raise ValueError(
                    f"Attribute '{attr_name}' must have the same length as 'id'."
                )
        return self


class BatchedQuestionAnswerPairSequence(BatchedBaseDataModel):
    """
    A class for representing a batch of sequences of question-answer pairs.

    Attributes:
        id (List[List[int]]): The unique identifiers for the question-answer pairs in each sequence.
        question_text (List[List[str]]): The list of question texts in each sequence.
        answer_start (List[List[int]]): The start positions of the answers in each sequence.
        answer_end (List[List[int]]): The end positions of the answers in each sequence.
        answer_text (List[List[str]]): The texts of the answers in each sequence.
        answer_is_correct (List[List[bool]]): Indicates whether each answer in each sequence is correct.
    """

    id: List[List[int]]
    question_text: List[List[str]]
    answer_start: List[List[int]]
    answer_end: List[List[int]]
    answer_text: List[List[str]]
    answer_is_correct: List[List[bool]]
