from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.document_content import DocumentContent
from atria_core.types.generic.image import Image
from atria_core.types.generic.ocr import OCR
from atria_core.types.generic.pdf import PDF
from pydantic import model_validator


class DocumentInstance(BaseDataInstance):
    page_id: int | None = None
    pdf: PDF | None = None
    image: Image | None = None
    ocr: OCR | None = None
    content: DocumentContent | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "DocumentInstance":
        from atria_core.types.ocr_parsers.hocr_parser import OCRProcessor

        # Ensure we have either image or PDF
        if self.image is None and self.pdf is None:
            raise ValueError("Either image or pdf must be provided")

        # Ensure we don't have both image and PDF
        if self.image is not None and self.pdf is not None:
            raise ValueError("Cannot have both image and pdf. Choose one.")

        if self.ocr is not None and self.content is None:
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

            self.content = OCRProcessor.parse(
                raw_ocr=self.ocr.content, ocr_type=self.ocr.type
            )

            if not was_loaded:
                self.ocr.unload()
        return self
