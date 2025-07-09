from typing import TYPE_CHECKING

import torch

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.generic._tensor.annotated_object import TensorAnnotatedObjectList
from atria_core.types.generic._tensor.bounding_box import TensorBoundingBoxList
from atria_core.types.generic._tensor.label import TensorLabel, TensorLabelList
from atria_core.types.generic._tensor.question_answer_pair import (
    TensorQuestionAnswerPair,
)

if TYPE_CHECKING:
    from atria_core.types.generic._raw.ground_truth import (
        OCRGT,  # noqa
        SERGT,  # noqa
        ClassificationGT,  # noqa
        GroundTruth,  # noqa
        LayoutAnalysisGT,  # noqa
        QuestionAnswerGT,  # noqa
        VisualQuestionAnswerGT,  # noqa
    )


class TensorClassificationGT(TensorDataModel["ClassificationGT"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.ClassificationGT"
    label: TensorLabel


class TensorOCRGT(TensorDataModel["OCRGT"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.OCRGT"
    words: list[str] | None = None
    word_bboxes: TensorBoundingBoxList | None = None
    word_confs: torch.Tensor | None = None
    word_angles: torch.Tensor | None = None


class TensorSERGT(TensorDataModel["SERGT"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.SERGT"
    words: list[str] | None = None
    word_bboxes: TensorBoundingBoxList | None = None
    word_labels: TensorLabelList | None = None
    segment_level_bboxes: TensorBoundingBoxList | None = None


class TensorQuestionAnswerGT(TensorDataModel["QuestionAnswerGT"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.QuestionAnswerGT"
    qa_pair: TensorQuestionAnswerPair
    words: list[str]


class TensorVisualQuestionAnswerGT(TensorDataModel["VisualQuestionAnswerGT"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.VisualQuestionAnswerGT"
    qa_pair: TensorQuestionAnswerPair
    words: list[str]
    word_bboxes: TensorBoundingBoxList
    segment_level_bboxes: TensorBoundingBoxList


class TensorLayoutAnalysisGT(TensorDataModel["LayoutAnalysisGT"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.LayoutAnalysisGT"
    annotated_objects: TensorAnnotatedObjectList | None = None
    words: list[str] | None = None
    word_bboxes: TensorBoundingBoxList | None = None


class TensorGroundTruth(TensorDataModel["GroundTruth"]):
    _raw_model = "atria_core.types.generic._raw.ground_truth.GroundTruth"
    classification: TensorClassificationGT | None = None
    ser: TensorSERGT | None = None
    ocr: TensorOCRGT | None = None
    qa: TensorQuestionAnswerGT | None = None
    vqa: TensorVisualQuestionAnswerGT | None = None
    layout: TensorLayoutAnalysisGT | None = None
