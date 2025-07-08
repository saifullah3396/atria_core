from typing import TYPE_CHECKING, ClassVar

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.generic._raw.ocr import OCRType

if TYPE_CHECKING:
    from atria_core.types.generic._raw.ocr import OCR  # noqa


class TensorOCR(TensorDataModel["OCR"]):
    _raw_model = "atria_core.types.generic._raw.ocr.OCR"
    _batch_merge_fields: ClassVar[list[str] | None] = ["type"]
    type: OCRType
    content: str
