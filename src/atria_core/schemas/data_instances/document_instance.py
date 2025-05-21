from enum import Enum

from atria_core.schemas.base import BaseDatabaseSchema, OptionalModel
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.ocr import OCRType


class OCRStatus(str, Enum):
    UNINITIATED = "uninitiated"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class DocumentInstanceBase(BaseDataInstance):
    doc_id: str
    page_id: int = 0
    total_num_pages: int = 1
    sample_path: str | None = None
    ocr_type: OCRType | None = None
    ocr_processing_status: OCRStatus = OCRStatus.UNINITIATED
    data: dict


class DocumentInstanceCreate(DocumentInstanceBase):
    split_id: SerializableUUID


class DocumentInstanceUpdate(OptionalModel):
    doc_id: str
    ground_truth: GroundTruth


class DocumentInstance(DocumentInstanceBase, BaseDatabaseSchema):
    split_id: SerializableUUID
