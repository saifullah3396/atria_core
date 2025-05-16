import enum


class TaskType(str, enum.Enum):
    image_classification = "image_classification"
    sequence_classification = "sequence_classification"
    sermantic_entity_recognition = "sermantic_entity_recognition"
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
