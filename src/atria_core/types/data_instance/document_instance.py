from pydantic import model_validator

from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image
from atria_core.types.generic.ocr import OCR
from atria_core.types.typing.common import IntField


class DocumentInstance(BaseDataInstance):
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
            if self.ocr.type is None:
                raise ValueError("OCR type not found after loading")

            self.gt.ocr = OCRProcessor.parse(
                raw_ocr=self.ocr.content, ocr_type=self.ocr.type
            )

            if not was_loaded:
                self.ocr.unload()
        return self
