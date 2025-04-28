import pytest
from data.structures.conftest import OCRFactory
from data.structures.tests_base import BaseDataModelTestBase

from atria.data.structures.base.data_model import BaseDataModel
from atria.data.structures.generic.ocr import BatchedOCR


class TestOCR(BaseDataModelTestBase):
    @pytest.fixture(
        params=[
            "from_file",
            "from_factory",
        ]
    )
    def backend(self, request):
        return request.param

    @pytest.fixture
    def model_instance(self, backend):
        OCRFactory.backend = backend
        return OCRFactory.build()

    def batched_model(self) -> type[BaseDataModel]:
        return BatchedOCR

    def tensor_fields(self) -> list[str]:
        return []

    # def test_invalid_ocr(self):
    #     with pytest.raises(ValueError):
    #         OCR().to_tensor()

    #     with pytest.raises(FileNotFoundError):
    #         OCR(
    #             file_path="test.hocr",
    #             raw_content=None,
    #             content=None,
    #             ocr_type=OCRType.TESSERACT,
    #         ).to_tensor()

    #     with pytest.raises(AssertionError):
    #         OCR(
    #             file_path=None,
    #             raw_content=MOCK_HOCR_TESSERACT,
    #             content=None,
    #         ).to_tensor()

    # def test_content(self, backend, model_instance):
    #     if backend in ["raw", "with_file"]:
    #         ocr_instance = model_instance.to_tensor()
    #         assert ocr_instance.raw_content == MOCK_HOCR_TESSERACT
    #         assert ocr_instance.content is not None
    #         assert ocr_instance.ocr_type == OCRType.TESSERACT
    #         assert ocr_instance.content.word_bboxes.shape == (7, 4)
    #         assert len(ocr_instance.content.words) == 7
    #         assert len(ocr_instance.content.word_confs) == 7
    #         assert len(ocr_instance.content.word_angles) == 7
    #     else:
    #         ocr_instance = model_instance.to_tensor()
    #         assert ocr_instance.raw_content is None
    #         assert ocr_instance.content is not None
    #         assert ocr_instance.ocr_type == OCRType.TESSERACT
    #         assert ocr_instance.content.word_bboxes.shape == (16, 4)
    #         assert len(ocr_instance.content.words) == 16
    #         assert len(ocr_instance.content.word_confs) == 16
    #         assert len(ocr_instance.content.word_angles) == 16
