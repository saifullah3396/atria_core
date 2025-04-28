"""
Data Structures Module

This module serves as the entry point for importing various data structures used in the system.
It provides access to foundational classes for data models, data instances, and generic structures
such as bounding boxes, images, labels, and OCR data.

Exports:
    - BaseDataModel: The base class for data models.
    - BaseDataInstance: The base class for individual data instances.
    - DocumentInstance: A class for representing documents with associated metadata and annotations.
    - ImageInstance: A class for representing images with associated metadata and annotations.
    - BoundingBox: A class for representing and manipulating bounding boxes.
    - Image: A class for representing and manipulating image data.
    - Label: A class for representing labels associated with data instances.
    - OCR: A class for managing OCR data.
    - OCRType: Enum for specifying the type of OCR system used.

Dependencies:
    - atria_core.data_types.base.data_model: For the `BaseDataModel` class.
    - atria_core.data_types.data_instance.base: For the `BaseDataInstance` class.
    - atria_core.data_types.data_instance.document: For the `DocumentInstance` class.
    - atria_core.data_types.data_instance.image: For the `ImageInstance` class.
    - atria_core.data_types.generic.bounding_box: For the `BoundingBox` class.
    - atria_core.data_types.generic.image: For the `Image` class.
    - atria_core.data_types.generic.label: For the `Label` class.
    - atria_core.data_types.generic.ocr: For the `OCR` class and `OCRType` enum.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "base.data_model",
        "data_instance.base",
        "data_instance.document",
        "data_instance.image",
        "generic.bounding_box",
        "generic.image",
        "generic.label",
        "generic.ocr",
    ],
    submod_attrs={
        "base.data_model": ["BaseDataModel"],
        "data_instance.base": ["BaseDataInstance"],
        "data_instance.document": ["DocumentInstance"],
        "data_instance.image": ["ImageInstance"],
        "generic.bounding_box": ["BoundingBox"],
        "generic.image": ["Image"],
        "generic.label": ["Label"],
        "generic.ocr": ["OCR", "OCRType"],
    },
)
