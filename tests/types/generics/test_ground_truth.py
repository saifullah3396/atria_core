import pyarrow as pa

from atria_core.types.factory import GroundTruthFactory
from tests.types.data_model_test_base import DataModelTestBase


class TestGroundTruth(DataModelTestBase):
    """
    Test class for OCR.
    """

    factory = GroundTruthFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "classification": pa.string(),
            "ser": pa.string(),
            "ocr": pa.string(),
            "qa": pa.string(),
            "vqa": pa.string(),
            "layout": pa.string(),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "classification": pa.string(),
            "ser": pa.string(),
            "ocr": pa.string(),
            "qa": pa.string(),
            "vqa": pa.string(),
            "layout": pa.string(),
        }
