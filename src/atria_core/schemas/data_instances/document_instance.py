from enum import Enum
from atria_core.schemas.base import BaseDatabaseSchema, DataInstanceType, OptionalModel
from atria_core.schemas.dataset import DatasetSplit
from atria_core.schemas.utils import SerializableUUID
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.annotated_object import AnnotatedObjectSequence
from atria_core.types.generic.label import Label
from atria_core.types.generic.ocr import OCRType
from atria_core.types.generic.question_answer_pair import QuestionAnswerPairSequence


class OCRStatus(str, Enum):
    UNINITIATED = "uninitiated"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class DocumentInstanceBase(BaseDataInstance):
    split: DatasetSplit
    image_file_path: str
    ocr_file_path: str | None = None
    ocr_type: OCRType | None = OCRType.other
    ocr_processing_status: OCRStatus = OCRStatus.UNINITIATED
    label: Label | None = None
    question_answer_pairs: QuestionAnswerPairSequence | None = None
    annotated_objects: AnnotatedObjectSequence | None = None


class DocumentInstanceCreate(DocumentInstanceBase):
    dataset_version_id: SerializableUUID


class DocumentInstanceUpdate(DocumentInstanceBase, OptionalModel):
    pass


class DocumentInstance(DocumentInstanceBase, BaseDatabaseSchema):
    dataset_version_id: SerializableUUID
