IMPORT_CHECK = [
    # datasets config
    "AtriaDatasetConfig",
    "AtriaHuggingfaceDatasetConfig",
    "AtriaHubDatasetConfig",
    # datasets metadata
    "DatasetShardInfo",
    "SplitInfo",
    "DatasetLabels",
    "DatasetMetadata",
    "DatasetStorageInfo",
    # datasets splits
    "SplitConfig",
    "DatasetSplitType",
    # instance types
    "DocumentInstance",
    "ImageInstance",
    # generic types
    "BoundingBox",
    "Image",
    "Label",
    "OCR",
    "OCRType",
    "GroundTruth",
    "OCRGT",
    "SERGT",
    "ClassificationGT",
    "LayoutAnalysisGT",
    "QuestionAnswerGT",
    "VisualQuestionAnswerGT",
    "AnnotatedObject",
    "QuestionAnswerPair",
    "TokenizedQuestionAnswerPair",
]


def test_imports():
    for name in IMPORT_CHECK:
        try:
            __import__("atria_core.types", fromlist=[name])
        except ImportError as e:
            raise ImportError(f"Failed to import {name}: {e}") from e
