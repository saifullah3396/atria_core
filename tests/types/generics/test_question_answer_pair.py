import pyarrow as pa

from atria_core.types.factory import (
    QuestionAnswerPairFactory,
    TokenizedQuestionAnswerPairFactory,
)
from tests.types.data_model_test_base import DataModelTestBase


class TestQuestionAnswerPair(DataModelTestBase):
    """
    Test class for QuestionAnswerPair.
    """

    factory = QuestionAnswerPairFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "id": pa.int64(),
            "question_text": pa.string(),
            "answer_start": pa.list_(pa.int64()),
            "answer_end": pa.list_(pa.int64()),
            "answer_text": pa.list_(pa.string()),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "id": pa.int64(),
            "question_text": pa.string(),
            "answer_start": pa.list_(pa.int64()),
            "answer_end": pa.list_(pa.int64()),
            "answer_text": pa.list_(pa.string()),
        }


class TestTokenizedQuestionAnswerPair(DataModelTestBase):
    """
    Test class for QuestionAnswerPair.
    """

    factory = TokenizedQuestionAnswerPairFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "answer_starts": pa.list_(pa.int64()),
            "answer_ends": pa.list_(pa.int64()),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "answer_starts": pa.list_(pa.int64()),
            "answer_ends": pa.list_(pa.int64()),
        }
