import pytest

from atria_core.types.base.data_model import BatchedBaseDataModel
from atria_core.types.data_instance.document import (
    BatchedDocumentInstance,
    DocumentInstance,
)
from atria_core.types.generic.image import Image
from atria_core.types.generic.label import Label
from atria_core.types.generic.ocr import OCRType
from tests.types.factory import DocumentInstanceFactory
from tests.types.tests_base import BaseDataModelTestBase


class TestDocumentInstance(BaseDataModelTestBase):
    @pytest.fixture
    def model_instance(self) -> DocumentInstance:
        return DocumentInstanceFactory.build()

    def tensor_fields(self) -> list[str]:
        return []

    def batched_model(self) -> type[BatchedBaseDataModel]:
        return BatchedDocumentInstance

    def test_initialization(self, model_instance: DocumentInstance):
        # Check if the image attribute is an instance of Image
        document_instance = model_instance.to_tensor()

        assert isinstance(
            document_instance.image, Image
        ), "Image attribute is not an instance of Image"

        # Check if the label attribute is an instance of Label
        assert isinstance(
            document_instance.label, Label
        ), "Label attribute is not an instance of Label"

        assert document_instance.image.content.shape == (
            3,
            256,
            256,
        ), "Image content shape mismatch"
        assert document_instance.image.shape == (3, 256, 256), "Image size mismatch"
        assert document_instance.ocr.ocr_type == OCRType.TESSERACT, "OCR type mismatch"
        assert len(document_instance.ocr.graph.word_bboxes) == 7
        assert len(document_instance.ocr.graph.words) == 7
        assert len(document_instance.ocr.graph.word_confs) == 7
        assert len(document_instance.ocr.graph.word_angles) == 7
