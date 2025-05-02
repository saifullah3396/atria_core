"""
Dataset Splits Module

This module defines classes and enumerations for managing dataset splits and their configurations.
It includes functionality for representing dataset splits (e.g., train, test, validation) and
configuring the generation of dataset splits with additional arguments.

Classes:
    - DatasetSplit: An enumeration representing the dataset splits (train, test, validation).
    - SplitConfig: A configuration class for dataset splits, including generation arguments.

Dependencies:
    - dataclasses.field: For defining default values for dataclass fields.
    - enum.Enum: For creating enumerations.
    - typing: For type annotations.
    - pydantic.BaseModel: For defining and validating data models.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from dataclasses import field
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class DatasetSplitType(str, Enum):
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
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)
