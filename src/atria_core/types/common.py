import enum


class DatasetSplitType(str, enum.Enum):
    """
    An enumeration representing the dataset splits.

    Attributes:
        train (str): Represents the training split of the dataset.
        test (str): Represents the testing split of the dataset.
        validation (str): Represents the validation split of the dataset.
    """

    train = "train"
    test = "test"
    validation = "validation"


class OCRType(str, enum.Enum):
    """
    Enum for OCR types.

    Attributes:
        TESSERACT (str): Tesseract OCR.
        EASY_OCR (str): EasyOCR.
        GOOGLE_VISION (str): Google Vision OCR.
        AWS_REKOGNITION (str): AWS Rekognition OCR.
        AZURE_OCR (str): Azure OCR.
        OTHER (str): Custom OCR implementation.
    """

    tesseract = "tesseract"
    easy_ocr = "easy_ocr"
    google_vision = "google_vision"
    aws_rekognition = "aws_rekognition"
    azure_ocr = "azure_ocr"
    custom = "custom"
    other = "other"


class TaskType(str, enum.Enum):
    image_classification = "image_classification"
    sequence_classification = "sequence_classification"
    semantic_entity_recognition = "semantic_entity_recognition"
    layout_entity_recognition = "layout_entity_recognition"
    question_answering = "question_answering"
    visual_question_answering = "visual_question_answering"
    layout_analysis = "layout_analysis"


class ModelType(str, enum.Enum):
    timm = "timm"
    torchvision = "torchvision"
    transformers_image_classification = "transformers/image_classification"
    transformers_sequence_classification = "transformers/sequence_classification"
    transformers_token_classification = "transformers/token_classification"
    transformers_question_answering = "transformers/question_answering"
    diffusers = "diffusers"
    mmdet = "mmdet"


class ConfigType(str, enum.Enum):
    batch_sampler = "batch_sampler"
    data_pipeline = "data_pipeline"
    dataset = "dataset"
    data_transform = "data_transform"
    dataset_splitter = "dataset_splitter"
    dataset_storage_manager = "dataset_storage_manager"
    engine = "engine"
    engine_step = "engine_step"
    lr_scheduler_factory = "lr_scheduler_factory"
    metric_factory = "metric_factory"
    model = "model"
    model_pipeline = "model_pipeline"
    optimizer_factory = "optimizer_factory"
    task_pipeline = "task_pipeline"


class TrainingStage:
    """
    Defines constants for various training stages.

    Attributes:
        train (str): Represents the training stage.
        validation (str): Represents the validation stage.
        test (str): Represents the testing stage.
        inference (str): Represents the inference stage.
        predict (str): Represents the prediction stage.
        visualization (str): Represents the visualization stage.

    Methods:
        get(name: str) -> Any: Retrieves the value of a training stage by its name.
    """

    train = "train"
    validation = "validation"
    test = "test"
    inference = "inference"
    predict = "predict"
    visualization = "visualization"


class GANStage:
    """
    Defines constants for GAN-specific training stages.

    Attributes:
        train_generator (str): Represents the stage for training the generator.
        train_discriminator (str): Represents the stage for training the discriminator.
    """

    train_generator = "train_gen"
    train_discriminator = "train_disc"
