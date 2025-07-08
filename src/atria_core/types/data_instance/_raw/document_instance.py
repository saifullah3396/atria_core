from typing import TYPE_CHECKING

from pydantic import model_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.data_instance._raw.base import BaseDataInstance
from atria_core.types.generic._raw.ground_truth import GroundTruth
from atria_core.types.generic._raw.image import Image
from atria_core.types.generic._raw.ocr import OCR
from atria_core.types.typing.common import IntField

if TYPE_CHECKING:
    from atria_core.types.data_instance._tensor.document_instance import (
        TensorDocumentInstance,  # noqa
    )


class DocumentInstance(BaseDataInstance, RawDataModel["TensorDocumentInstance"]):  # type: ignore[misc]
    _tensor_model = "atria_core.types.data_instance._tensor.document_instance.TensorDocumentInstance"
    page_id: IntField = 0
    total_num_pages: IntField = 1
    image: Image
    ocr: OCR | None = None
    gt: GroundTruth = GroundTruth()

    @model_validator(mode="after")
    def validate_fields(self) -> "DocumentInstance":
        from atria_core.types.ocr_parsers.hocr_parser import OCRProcessor

        if self.image is None and self.ocr is None:
            raise ValueError("At least one of image or ocr must be provided")

        if self.ocr is not None and self.gt.ocr is None:
            # here we load the ocr content if it is not already loaded
            # in order to parse it into its ground truth format
            # this is necessary because the ocr content will not be serialized if required for the table
            # but the ground truth will be serialized into binary json format
            was_loaded = self.ocr.content is not None
            if not was_loaded:
                self.ocr.load()

            if self.ocr.content is None:
                raise ValueError("OCR content not found after loading")

            self.gt.ocr = OCRProcessor.parse(
                raw_ocr=self.ocr.content, ocr_type=self.ocr.type
            )

            if not was_loaded:
                self.ocr.unload()
        return self
