from unittest.mock import MagicMock, patch

import pyarrow as pa

from atria_core.types.factory import MOCK_HOCR_TESSERACT, OCRFactory
from atria_core.types.generic._raw.ocr import OCR
from tests.types.data_model_test_base import DataModelTestBase


class TestOCR(DataModelTestBase):
    """
    Test class for OCR.
    """

    factory = OCRFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"file_path": pa.string(), "type": pa.string(), "content": pa.binary()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"file_path": pa.string(), "type": pa.string(), "content": pa.binary()}


#########################################################
# Basic OCR Tests
#########################################################
@patch("requests.get")
def test_load_from_url(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = MOCK_HOCR_TESSERACT.encode("utf-8")
    mock_get.return_value = mock_response

    raw_image = OCR(file_path="https://example.com/test_image.txt")
    raw_image.load()
    assert raw_image.content is not None
