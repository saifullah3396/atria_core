import pytest

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.ocr import BatchedOCR
from tests.types.factory import OCRFactory
from tests.types.tests_base import BaseDataModelTestBase


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
