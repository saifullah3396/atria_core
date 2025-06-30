"""
Dataset Configuration Module

This module defines configuration classes for datasets used in the Atria application.
These configurations are used to standardize and manage metadata, data sources, and
other properties of datasets.

Classes:
    - AtriaDatasetConfig: Base configuration class for datasets.
    - AtriaHuggingfaceDatasetConfig: Configuration class for Hugging Face datasets.

Dependencies:
    - pydantic.BaseModel: For defining and validating configuration models.
    - typing: For type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict


class AtriaDatasetConfig(BaseModel):
    """
    Base configuration class for datasets.

    This class provides a standard structure for defining dataset configurations,
    including metadata such as name, description, version, and data URLs.

    Attributes:
        name (str): The name of the dataset.
        config_name (str): The name of the dataset.
        data_urls (Union[str, List[str], Dict[str, str]] | None): The URLs for accessing
            the dataset. Can be a single URL, a list of URLs, or a dictionary mapping
            keys to URLs. Defaults to None.
        max_train_samples (Optional[int]): The maximum number of training samples to load.
        max_validation_samples (Optional[int]): The maximum number of validation samples to load.
        max_test_samples (Optional[int]): The maximum number of test samples to load.
    """

    model_config = ConfigDict(extra="forbid")

    __target__: str = "atria.data.dataset.atria_dataset.AtriaDataset"
    data_urls: Union[str, List[str], Dict[str, str], Dict[str, Tuple]] | None = None
    max_train_samples: Optional[int] = None
    max_validation_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


class AtriaHuggingfaceDatasetConfig(AtriaDatasetConfig):
    """
    Configuration class for Hugging Face datasets.

    This class extends `AtriaDatasetConfig` to include additional properties specific
    to datasets hosted on Hugging Face.

    Attributes:
        hf_repo (str): The Hugging Face repository name for the dataset.
    """

    __target__: str = (
        "atria.data.dataset.atria_huggingface_dataset.AtriaHuggingfaceDataset"
    )
    hf_repo: str


class AtriaHubDatasetConfig(BaseModel):
    """
    Configuration class for Hub datasets.

    This class extends `AtriaDatasetConfig` to include additional properties specific
    to datasets hosted on Hub.
    """

    __target__: str = "atria.hub.atria_hub_dataset.AtriaHubDataset"
