IMPORT_CHECK = [
    # common types
    "ConfigType",
    "DatasetSplitType",
    "GANStage",
    "ModelType",
    "OCRType",
    "TaskType",
    "TrainingStage",
    # data instance types
    "BaseDataInstance",
    "DocumentInstance",
    "ImageInstance",
    # datasets metadata
    "DatasetLabels",
    "DatasetMetadata",
    "DatasetShardInfo",
    "SplitConfig",
    "SplitInfo",
    # generic types
    "AnnotatedObject",
    "AnnotatedObjectList",
    "BoundingBox",
    "BoundingBoxList",
    "BoundingBoxMode",
    "DocumentContent",
    "Annotation",
    "ClassificationAnnotation",
    "EntityLabelingAnnotation",
    "LayoutAnalysisAnnotation",
    "ExtractiveQAAnnotation",
    "GenerativeQAAnnotation",
    "Image",
    "Label",
    "LabelList",
    "OCR",
    "ExtractiveQAPair",
    "GenerativeQAItem",
]


def test_imports():
    for name in IMPORT_CHECK:
        module = __import__("atria_core.types", fromlist=[name])
        assert hasattr(module, name), f"Cannot import {name} from atria_core.types"
