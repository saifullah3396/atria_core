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

from typing import Dict, List, Union

from pydantic import BaseModel

from atria_core.constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR


class AtriaDatasetConfig(BaseModel):
    """
    Base configuration class for datasets.

    This class provides a standard structure for defining dataset configurations,
    including metadata such as name, description, version, and data URLs.

    Attributes:
        config_name (str): The name of the dataset.
        data_urls (Union[str, List[str], Dict[str, str]] | None): The URLs for accessing
            the dataset. Can be a single URL, a list of URLs, or a dictionary mapping
            keys to URLs. Defaults to None.
        streaming_mode (bool): Indicates whether the dataset should be loaded in
    """

    __target__: str = "atria.data.dataset.atria_dataset.AtriaDataset"
    data_dir: str = _DEFAULT_ATRIA_DATASETS_CACHE_DIR
    data_urls: Union[str, List[str], Dict[str, str]] | None = None
    streaming_mode: bool = False


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


class AtriaHubDatasetConfig(AtriaDatasetConfig):
    """
    Configuration class for Hub datasets.

    This class extends `AtriaDatasetConfig` to include additional properties specific
    to datasets hosted on Hub.

    Attributes:
        hf_repo (str): The Hub repository name for the dataset.
    """

    __target__: str = (
        "atria.data.dataset.atria_huggingface_dataset.AtriaHuggingfaceDataset"
    )
    hf_repo: str
