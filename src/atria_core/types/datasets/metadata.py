import json
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from rich.pretty import pretty_repr

from atria_core.logger import get_logger
from atria_core.types.common import DatasetSplitType
from atria_core.utilities.repr import RepresentationMixin

if TYPE_CHECKING:
    from datasets.info import DatasetInfo  # type: ignore[import-not-found]

logger = get_logger(__name__)


class SplitConfig(BaseModel):
    """
    A configuration class for dataset splits.

    This class defines the split type and additional keyword arguments for generating
    the dataset split.

    Attributes:
        split (DatasetSplit): The type of dataset split (e.g., train, test, validation).
        gen_kwargs (Dict[str, Any]): Additional keyword arguments for generating the split.
    """

    split: DatasetSplitType
    gen_kwargs: dict[str, Any] = field(default_factory=dict)


class DatasetShardInfo(BaseModel):
    """
    Represents information about a dataset shard.

    Attributes:
        url (str): The URL of the shard.
        shard (int): The shard number.
        total (int): The total number of shards.
        nsamples (int): The number of examples in the shard.
        filesize (int): The size of the shard in bytes.
    """

    url: str = ""
    shard: int = 1
    nsamples: int = 0
    filesize: int = 0

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
    shardlist: list[DatasetShardInfo]

    @classmethod
    def from_shard_info_list(cls, shard_list: list[DatasetShardInfo]) -> "SplitInfo":
        """
        Creates a SplitInfo instance from a list of DatasetShardInfo.

        Args:
            shard_list (List[DatasetShardInfo]): A list of shard information.

        Returns:
            SplitInfo: The created SplitInfo instance.
        """
        num_bytes = sum(shard.filesize for shard in shard_list)
        num_examples = sum(shard.nsamples for shard in shard_list)
        return cls(num_bytes=num_bytes, num_examples=num_examples, shardlist=shard_list)

    def __repr__(self):
        return pretty_repr(self)

    def __str__(self):
        return super().__repr__()

    def to_file(self, file_path: Path | str):
        """
        Serializes and saves the storage information to a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.
        """
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: Path | str):
        """
        Loads storage information from a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.

        Returns:
            DatasetStorageInfo: The loaded storage information.
        """
        with Path(file_path).open("r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.model_validate_json(json_data, strict=True)


class DatasetLabels(BaseModel):
    """
    Represents classification and token labels for a dataset.

    Attributes:
        classification (List[str] | None): The classification labels.
        ser (List[str] | None): The semantic entity recognition labels.
        layout (List[str] | None): The layout labels.
    """

    classification: list[str] | None = None
    ser: list[str] | None = None
    layout: list[str] | None = None

    @classmethod
    def _infer_from_huggingface_features(cls, features) -> "DatasetLabels":
        """
        Infers labels from Hugging Face dataset features.

        Args:
            features: The dataset features.

        Returns:
            DatasetLabels: The inferred labels.
        """
        import datasets  # type: ignore[import-not-found]

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
                for _, obj_value in value[0].items():
                    if isinstance(obj_value, datasets.ClassLabel):
                        object_labels = obj_value.names
        if instance_labels is None and object_labels is None and token_labels is None:
            logger.warning(
                "No labels found in the dataset features. Please check the dataset structure."
            )
        return cls(
            classification=instance_labels, layout=object_labels, ser=token_labels
        )

    def __repr__(self):
        return pretty_repr(self)

    def __str__(self):
        return super().__repr__()


class DatasetMetadata(BaseModel, RepresentationMixin):  # type: ignore[misc]
    """
    Represents metadata for a dataset, including configuration and labels.

    Attributes:
        homepage (str | None): The homepage URL of the dataset.
        description (str | None): A description of the dataset.
        license (str | None): The license of the dataset.
        citation (str | None): Citation information for the dataset.
        dataset_labels (DatasetLabels): The labels associated with the dataset.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    homepage: str | None = None
    description: str | None = None
    license: str | None = None
    citation: str | None = None
    dataset_labels: DatasetLabels = DatasetLabels()

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
            DatasetMetadata: The loaded metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if Path(file_path).exists():
            with open(file_path, encoding="utf-8") as f:
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
            DatasetMetadata: The created metadata.
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


class DatasetStorageInfo(BaseModel, RepresentationMixin):  # type: ignore[misc]
    """
    Represents storage information for a dataset, including metadata and split info.

    Attributes:
        metadata (DatasetMetadata): The metadata for the dataset.
        split_info (SplitInfo): The split information for the dataset.
    """

    metadata: DatasetMetadata
    split_info: SplitInfo

    def to_file(self, file_path: Path | str):
        """
        Serializes and saves the storage information to a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.
        """
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: Path | str):
        """
        Loads storage information from a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.

        Returns:
            DatasetStorageInfo: The loaded storage information.
        """
        with Path(file_path).open("r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.model_validate_json(json_data, strict=True)
