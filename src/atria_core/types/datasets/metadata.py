"""
Dataset Metadata Module

This module defines classes for managing metadata and related information for datasets
used in the Atria application. It includes functionality for handling dataset splits,
labels, storage information, and serialization/deserialization of metadata.

Classes:
    - DatasetShardInfo: Represents information about a dataset shard.
    - SplitInfo: Represents information about a dataset split.
    - DatasetLabels: Represents classification and token labels for a dataset.
    - AtriaDatasetMetadata: Represents metadata for a dataset, including configuration and labels.
    - AtriaDatasetStorageInfo: Represents storage information for a dataset, including metadata and split info.

Dependencies:
    - pydantic.BaseModel: For defining and validating data models.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations and generic types.
    - rich.pretty: For generating rich representations of objects.
    - atria_types.data_instance.base: For base data instance structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from rich.pretty import pretty_repr

from atria_core.logger import get_logger
from atria_core.types.datasets.config import AtriaDatasetConfig
from atria_core.utilities.repr import RepresentationMixin

if TYPE_CHECKING:
    from atria_types.data_instance.base import BaseDataInstance
    from datasets import DatasetInfo

logger = get_logger(__name__)


class DatasetShardInfo(BaseModel):
    """
    Represents information about a dataset shard.

    Attributes:
        url (str): The URL of the shard.
        shard (int): The shard number.
        total (int): The total number of shards.
        count (int): The number of examples in the shard.
        size (int): The size of the shard in bytes.
    """

    url: str = ""
    shard: int = 1
    total: int = 0
    count: int = 0
    size: int = 0

    def __repr__(self):
        return pretty_repr(self)

    def __str__(self):
        return super().__repr__()


class SplitInfo(BaseModel):
    """
    Represents information about a dataset split.

    Attributes:
        num_bytes (int): The total size of the split in bytes.
        num_examples (int): The total number of examples in the split.
        shards (List[DatasetShardInfo]): A list of shard information for the split.
    """

    num_bytes: int
    num_examples: int
    shards: List[DatasetShardInfo]

    @classmethod
    def from_shard_info_list(cls, shard_list: List[DatasetShardInfo]) -> "SplitInfo":
        """
        Creates a SplitInfo instance from a list of DatasetShardInfo.

        Args:
            shard_list (List[DatasetShardInfo]): A list of shard information.

        Returns:
            SplitInfo: The created SplitInfo instance.
        """
        num_bytes = sum(shard.size for shard in shard_list)
        num_examples = sum(shard.count for shard in shard_list)
        return cls(num_bytes=num_bytes, num_examples=num_examples, shards=shard_list)

    def __repr__(self):
        return pretty_repr(self)

    def __str__(self):
        return super().__repr__()


class DatasetLabels(BaseModel):
    """
    Represents classification and token labels for a dataset.

    Attributes:
        instance_classification (Optional[List[str]]): Labels for instance-level classification.
        object_classification (Optional[List[str]]): Labels for object-level classification.
        token_classification (Optional[List[str]]): Labels for token-level classification.
    """

    instance_classification: Optional[List[str]] = None
    object_classification: Optional[List[str]] = None
    token_classification: Optional[List[str]] = None

    @classmethod
    def _infer_from_huggingface_features(cls, features) -> "DatasetLabels":
        """
        Infers labels from Hugging Face dataset features.

        Args:
            features: The dataset features.

        Returns:
            DatasetLabels: The inferred labels.
        """
        import datasets

        instance_labels = None
        object_labels = None
        token_labels = None
        for key, value in features.items():
            if isinstance(value, datasets.ClassLabel):
                instance_labels = value.names
            elif isinstance(value, datasets.Sequence) and isinstance(
                value.feature, datasets.ClassLabel
            ):
                token_labels = value.feature.names
            elif isinstance(value, list) and "objects" in key:
                for obj_key, obj_value in value[0].items():
                    if isinstance(obj_value, datasets.ClassLabel):
                        object_labels = obj_value.names
        if instance_labels is None and object_labels is None and token_labels is None:
            logger.warning(
                "No labels found in the dataset features. Please check the dataset structure."
            )
        return cls(
            instance_classification=instance_labels,
            object_classification=object_labels,
            token_classification=token_labels,
        )

    def __repr__(self):
        return pretty_repr(self)

    def __str__(self):
        return super().__repr__()


class AtriaDatasetMetadata(BaseModel, RepresentationMixin):
    """
    Represents metadata for a dataset, including configuration and labels.

    Attributes:
        dataset_name (str | None): The name of the dataset.
        citation (str | None): The citation for the dataset.
        homepage (str | None): The homepage URL for the dataset.
        license (str | None): The license for the dataset.
        config (AtriaDatasetConfig | None): The configuration for the dataset.
        dataset_labels (DatasetLabels | None): The labels for the dataset.
        data_model (Type[BaseDataInstance] | None): The data model class for the dataset.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    dataset_name: str | None = None
    citation: str | None = None
    homepage: str | None = None
    license: str | None = None
    config: AtriaDatasetConfig | None = None
    dataset_labels: DatasetLabels | None = None
    data_model: Type[Any] | None = None

    @field_serializer("data_model")
    def serialize_data_model(self, data_model: Type["BaseDataInstance"], _info) -> str:
        """
        Serializes the data model to a string.

        Args:
            data_model (Type[BaseDataInstance]): The data model class.

        Returns:
            str: The serialized data model.
        """
        return f"{data_model.__module__}.{data_model.__name__}"

    @field_validator("data_model", mode="before")
    @classmethod
    def validate_data_model(cls, value: str) -> Type["BaseDataInstance"]:
        """
        Validates and deserializes the data model from a string.

        Args:
            data_model (str): The serialized data model.

        Returns:
            Type[BaseDataInstance]: The deserialized data model class.
        """
        if isinstance(value, str):
            module_name, class_name = value.rsplit(".", 1)
            module = importlib.import_module(module_name)
            value = getattr(module, class_name)
        return value

    def to_file(self, file_path: str):
        """
        Serializes and saves the metadata to a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: str):
        """
        Loads metadata from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            AtriaDatasetMetadata: The loaded metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if Path(file_path).exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        else:
            raise FileNotFoundError(f"Dataset info file not found at {file_path}")

    @classmethod
    def from_huggingface_info(cls, info: "DatasetInfo"):
        """
        Creates metadata from Hugging Face dataset info.

        Args:
            info (DatasetInfo): The Hugging Face dataset info.

        Returns:
            AtriaDatasetMetadata: The created metadata.
        """
        return cls(
            citation=info.citation,
            homepage=info.homepage,
            license=info.license,
            dataset_labels=DatasetLabels._infer_from_huggingface_features(
                info.features
            ),
        )

    def state_dict(self):
        """
        Get the state dictionary for the dataset metadata.

        Returns:
            Dict: A dictionary containing the state of the dataset metadata.
        """
        return self.model_dump()

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary for the dataset metadata.

        Returns:
            None
        """
        self.model_validate(state_dict)


class AtriaDatasetStorageInfo(BaseModel, RepresentationMixin):
    """
    Represents storage information for a dataset, including metadata and split info.

    Attributes:
        metadata (AtriaDatasetMetadata): The metadata for the dataset.
        split_info (SplitInfo): The split information for the dataset.
    """

    metadata: AtriaDatasetMetadata
    split_info: SplitInfo

    def to_file(self, file_path: Union[Path, str]):
        """
        Serializes and saves the storage information to a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.
        """
        logger.info(
            f"Writing dataset storage info to {file_path}:\n{pretty_repr(self)}."
        )
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        """
        Loads storage information from a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.

        Returns:
            AtriaDatasetStorageInfo: The loaded storage information.
        """
        with Path(file_path).open("r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.model_validate_json(json_data, strict=True)
