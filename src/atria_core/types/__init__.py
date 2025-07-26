# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy


from .model_outputs.outputs import *

if TYPE_CHECKING:
    from .common import (
        ConfigType,
        DatasetSplitType,
        GANStage,
        ModelType,
        OCRType,
        TaskType,
        TrainingStage,
    )
    from .data_instance.base import BaseDataInstance
    from .data_instance.document_instance import DocumentInstance
    from .data_instance.image_instance import ImageInstance
    from .datasets.metadata import (
        DatasetLabels,
        DatasetMetadata,
        DatasetShardInfo,
        SplitConfig,
        SplitInfo,
    )
    from .generic.annotated_object import AnnotatedObject, AnnotatedObjectList
    from .generic.bounding_box import BoundingBox, BoundingBoxList, BoundingBoxMode
    from .generic.ground_truth import (
        OCRGT,
        SERGT,
        ClassificationGT,
        GroundTruth,
        LayoutAnalysisGT,
        QuestionAnswerGT,
        VisualQuestionAnswerGT,
    )
    from .generic.image import Image
    from .generic.label import Label, LabelList
    from .generic.ocr import OCR
    from .generic.question_answer_pair import QuestionAnswerPair

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "common": [
            "ConfigType",
            "DatasetSplitType",
            "GANStage",
            "ModelType",
            "OCRType",
            "TaskType",
            "TrainingStage",
        ],
        "data_instance.base": ["BaseDataInstance"],
        "data_instance.document_instance": ["DocumentInstance"],
        "data_instance.image_instance": ["ImageInstance"],
        "datasets.metadata": [
            "DatasetLabels",
            "DatasetMetadata",
            "DatasetShardInfo",
            "SplitConfig",
            "SplitInfo",
        ],
        "generic.annotated_object": ["AnnotatedObject", "AnnotatedObjectList"],
        "generic.bounding_box": ["BoundingBox", "BoundingBoxList", "BoundingBoxMode"],
        "generic.ground_truth": [
            "OCRGT",
            "SERGT",
            "ClassificationGT",
            "GroundTruth",
            "LayoutAnalysisGT",
            "QuestionAnswerGT",
            "VisualQuestionAnswerGT",
        ],
        "generic.image": ["Image"],
        "generic.label": ["Label", "LabelList"],
        "generic.ocr": ["OCR"],
        "generic.question_answer_pair": ["QuestionAnswerPair"],
    },
)
