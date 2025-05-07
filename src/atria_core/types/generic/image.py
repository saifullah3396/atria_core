"""
Image Module

This module defines the `Image` and `BatchedImage` classes, which represent images and their batched counterparts.
It includes functionality for loading image data, validating and serializing image tensors, and computing properties
such as shape, size, width, height, and data type.

Classes:
    - Image: A class for representing and manipulating image data.
    - BatchedImage: A class for handling batched image data.

Dependencies:
    - pathlib.Path: For handling file paths.
    - PIL.Image: For loading image files.
    - torch: For tensor operations.
    - pydantic: For data validation and serialization.
    - atria_core.logger: For logging utilities.
    - atria_core.data_types.base.data_model: For the base data model class.
    - atria_core.data_types.typing.common: For common type annotations.
    - atria_core.data_types.typing.image: For image-specific type annotations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel, BaseDataModelConfigDict
from atria_core.types.typing.common import PydanticFilePath
from atria_core.utilities.encoding import _bytes_to_image, _image_to_bytes
from PIL.Image import Image as PILImage
from pydantic import field_serializer, field_validator, model_validator

logger = get_logger(__name__)


class Image(BaseDataModel):
    """
    A class for representing and manipulating image data.

    This class provides functionality for loading image data from a file or tensor,
    computing properties such as shape, size, width, height, and data type, and
    serializing image metadata.

    Attributes:
        file_path (PydanticFilePath | None): The file path to the image. Defaults to None.
        content (Optional[Union[torch.Tensor, PILImage]]): The image data as a tensor. Defaults to None.
        source_size (Tuple[int, int] | None): The original shape of the image. Defaults to None.
    """

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["file_path", "source_size"],
    )

    file_path: PydanticFilePath | None = None
    content: Optional[Union[torch.Tensor, PILImage]] = None
    source_size: Tuple[int, int] | None = None

    @model_validator(mode="after")
    def load_content(self):
        """
        Loads the image data from the file path or validates the content.

        If the `file_path` is provided, the image is loaded from the file and converted
        to a tensor. If the `content` is already provided, it is validated.

        Raises:
            ValueError: If neither `file_path` nor `content` is provided.
            FileNotFoundError: If the file specified by `file_path` does not exist.
        """
        import PIL

        if self.file_path is None and self.content is None:
            raise ValueError("Either file_path or content must be provided.")
        if self.content is None:
            assert self.file_path is not None, "Image file path is not set."
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Image file not found: {self.file_path}")
            self.content = PIL.Image.open(self.file_path)
            self.source_size = self.content.size
        return self

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> torch.Tensor:
        """
        Validates and converts the input value to a tensor.

        Args:
            value (Any): The input value to validate.

        Returns:
            torch.Tensor: The validated tensor.

        Raises:
            AssertionError: If the input value is invalid.
        """
        import PIL
        from torchvision.transforms.functional import to_tensor

        if value is None:
            return value
        if isinstance(value, (bytes, str)):
            value = _bytes_to_image(value)
        elif isinstance(value, np.ndarray):
            try:
                value = PIL.Image.fromarray(value)
            except Exception as e:
                value = to_tensor(value)
        return value

    @field_serializer("content")
    def serialize_content(
        self, content: Optional[Union[torch.Tensor, PILImage]], _info
    ) -> bytes:
        """
        Serializes the image tensor to a base64-encoded string.

        Args:
            content (Optional[Union[torch.Tensor, PILImage]]): The image tensor to serialize.
            _info: Additional information (unused).

        Returns:
            str: The serialized image as a base64-encoded string.
        """
        if content is None:
            return content
        return _image_to_bytes(content)

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_shapes(cls, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            assert value.ndim in [
                2,
                3,
            ], f"Invalid number of dimensions in the image tensor: {value.shape}. Image tensor must be 2D (grayscale) or 3D (channels, height, width)."
            if value.ndim == 2:
                value = value.unsqueeze(0)
        return value

    def to_tensor(self) -> torch.Tensor:
        """
        Converts the image content to a tensor.

        Returns:
            torch.Tensor: The image content as a tensor.

        Raises:
            ValueError: If the image content is not loaded.
        """
        from torchvision.transforms.functional import to_tensor

        if not self._is_tensor or self._is_tensor is None:
            logger.debug(f"Converting {self.__class__.__name__} to tensors.")
            if isinstance(self.content, PILImage):
                self.content = to_tensor(self.content)
            self._is_tensor = True
        return self

    def from_tensor(self) -> None:
        """
        Converts a tensor to the image content.

        Args:
            tensor (torch.Tensor): The input tensor to convert.

        Raises:
            ValueError: If the input tensor is invalid.
        """
        from torchvision.transforms.functional import to_pil_image

        if self._is_tensor or self._is_tensor is None:
            logger.debug(f"Converting {self.__class__.__name__} from tensors.")
            if isinstance(self.content, torch.Tensor):
                self.content = to_pil_image(self.content)
            self._is_tensor = False
        return self

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the image tensor.

        Returns:
            torch.Size: The shape of the image tensor.
        """

        if isinstance(self.content, torch.Tensor):
            return self.content.shape
        elif isinstance(self.content, PILImage):
            return (len(self.content.getbands()), *self.content.size)
        return self.content.size

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the image tensor.

        Returns:
            torch.dtype: The data type of the image tensor.
        """
        if isinstance(self.content, torch.Tensor):
            return self.content.dtype

    @property
    def size(self) -> torch.Size:
        """
        Returns the size of the image as (width, height).

        Returns:
            torch.Size: The size of the image.
        """
        if isinstance(self.content, torch.Tensor):
            return (self.content.shape[2], self.content.shape[1])
        elif isinstance(self.content, PILImage):
            return self.content.size

    @property
    def width(self) -> int:
        """
        Returns the width of the image.

        Returns:
            int: The width of the image.
        """
        return self.size[0]

    @property
    def height(self) -> int:
        """
        Returns the height of the image.

        Returns:
            int: The height of the image.
        """
        return self.size[1]
