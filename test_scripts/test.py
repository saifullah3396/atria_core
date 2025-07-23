# # ruff: noqa
# from atria_core.types import Label, LabelList

# label = Label(value=0, name="Test Label")
# # # label_list = LabelList(value=[1, 2], name=["Label One", "Label Two"])

# # print(label)
# # print(label_list)

from atria_core.types.common import (
    ConfigType,
    DatasetSplitType,
    GANStage,
    ModelType,
    OCRType,
    TaskType,
    TrainingStage,
)
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.data_instance.document_instance import DocumentInstance
from atria_core.types.data_instance.image_instance import ImageInstance
from atria_core.types.datasets.config import (
    AtriaDatasetConfig,
    AtriaHubDatasetConfig,
    AtriaHuggingfaceDatasetConfig,
)
from atria_core.types.datasets.metadata import (
    DatasetLabels,
    DatasetMetadata,
    DatasetShardInfo,
    SplitConfig,
    SplitInfo,
)
from atria_core.types.generic.annotated_object import (
    AnnotatedObject,
    AnnotatedObjectList,
)
from atria_core.types.generic.bounding_box import (
    BoundingBox,
    BoundingBoxList,
    BoundingBoxMode,
)
from atria_core.types.generic.ground_truth import (
    OCRGT,
    SERGT,
    ClassificationGT,
    GroundTruth,
    LayoutAnalysisGT,
    QuestionAnswerGT,
    VisualQuestionAnswerGT,
)
from atria_core.types.generic.image import Image
from atria_core.types.generic.label import Label, LabelList
from atria_core.types.generic.ocr import OCR
from atria_core.types.generic.question_answer_pair import QuestionAnswerPair

import numpy as np

tensor_image = Image(content=np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
print(tensor_image)  # Should print (3, 32, 32)
assert tensor_image.shape == (3, 32, 32)
