"""
Constants Module

This module defines constants used across the `atria` package. These constants
include default paths and environment variable configurations for managing
resources such as models and file storage.

Constants:
    - _DEFAULT_ATRIA_CACHE_DIR: The default directory for caching Atria resources.
    - _DEFAULT_ATRIA_MODELS_CACHE_DIR: The default directory for caching Atria models.
    - _DEFAULT_ATRIA_FILE_STORAGE_DIR: The default directory for Atria file storage.

Dependencies:
    - os: For accessing environment variables.
    - pathlib: For handling file system paths.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import os
from pathlib import Path

DEFAULT_ATRIA_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", Path.home() / ".cache/atria/"
)
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_DATASETS_CACHE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "datasets/"
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_MODELS_CACHE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "models/"
"""
Path: The default directory for caching Atria models. This is a subdirectory of
`_DEFAULT_ATRIA_CACHE_DIR` and is used to store model-related resources.
"""

_DEFAULT_ATRIA_FILE_STORAGE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "fs/"
"""
Path: The default directory for Atria file storage. This is a subdirectory of
`_DEFAULT_ATRIA_CACHE_DIR` and is used to store file system-related resources.
"""

_DEFAULT_OPTIMIZER_PARAMETERS_KEY = "default"
"""
str: The default key for optimizer parameters in configuration. This key is used
to identify the default set of parameters for optimizers when configuring
optimizers in the Atria framework.
"""

_MAX_REPR_PRINT_ELEMENTS = 4
"""
int: The maximum number of elements to print in the string representation of
a collection. This is used to limit the output length when displaying
large collections in the Atria framework.
"""

_TORCH_PRINT_OPTIONS_PROFILE = "short"
"""
str: The default profile for PyTorch tensor printing options. This is used
to set the printing profile for PyTorch tensors in the Atria framework.
This profile determines how tensors are displayed when printed, including
the number of elements shown and the formatting style.
"""
