import enum


class TaskType(str, enum.Enum):
    classification = "classification"
    sequence_classification = "sequence_classification"
    sermantic_entity_recognition = "sermantic_entity_recognition"
    layout_analysis = "layout_analysis"
    question_answering = "question_answering"


class ModelType(str, enum.Enum):
    timm = "timm"
    torchvision = "torchvision"
    transformers_image_classification = "transformers/image_classification"
    transformers_question_answering = "transformers/question_answering"
    transformers_token_classification = "transformers/token_classification"
    transformers_sequence_classification = "transformers/sequence_classification"
    diffusers = "diffusers"
    mmdet = "mmdet"
    custom = "custom"
