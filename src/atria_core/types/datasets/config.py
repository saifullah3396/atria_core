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


class AtriaDatasetConfig(BaseModel):
    """
    Base configuration class for datasets.

    This class provides a standard structure for defining dataset configurations,
    including metadata such as name, description, version, and data URLs.

    Attributes:
        name (str): The name of the dataset.
        description (str | None): A brief description of the dataset. Defaults to None.
        version (str): The version of the dataset. Defaults to "0.0.0".
        data_urls (Union[str, List[str], Dict[str, str]] | None): The URLs for accessing
            the dataset. Can be a single URL, a list of URLs, or a dictionary mapping
            keys to URLs. Defaults to None.
    """

    name: str
    description: str | None = None
    version: str = "0.0.0"
    data_urls: Union[str, List[str], Dict[str, str]] | None = None


class AtriaHuggingfaceDatasetConfig(AtriaDatasetConfig):
    """
    Configuration class for Hugging Face datasets.

    This class extends `AtriaDatasetConfig` to include additional properties specific
    to datasets hosted on Hugging Face.

    Attributes:
        hf_repo (str): The Hugging Face repository name for the dataset.
    """

    hf_repo: str
