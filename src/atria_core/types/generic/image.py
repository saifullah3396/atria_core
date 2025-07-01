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
from typing import Any, ClassVar

import numpy as np
import torch
from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import (
    BaseDataModel,
    BaseDataModelConfigDict,
    RowSerializable,
)
from atria_core.types.typing.common import PydanticFilePath
from atria_core.utilities.encoding import (
    _base64_to_image,
    _bytes_to_image,
    _image_to_base64,
)
from PIL import Image as PILImageModule
from PIL.Image import Image as PILImage
from pydantic import field_serializer, field_validator, model_validator

logger = get_logger(__name__)


class Image(BaseDataModel, RowSerializable):
    """
    A class for representing and manipulating image data.

    This class provides functionality for loading image data from a file or tensor,
    computing properties such as shape, size, width, height, and data type, and
    serializing image metadata.

    Attributes:
        file_path (PydanticFilePath | None): The file path to the image. Defaults to None.
        content (Optional[Union[torch.Tensor, PILImage]]): The image data as a tensor. Defaults to None.
        width (int | None): The width of the image. Computed property.
        height (int | None): The height of the image. Computed property.
    """

    _row_name: ClassVar[str | None] = "image"
    _row_serialization_types: ClassVar[dict[str, str]] = {
        "file_path": str,
        "width": int,
        "height": int,
        "content": str,
    }

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["file_path"],
    )

    file_path: PydanticFilePath | None = None
    content: torch.Tensor | PILImage | None = None
    width: int | None = None
    height: int | None = None

    @model_validator(mode="after")
    def check_dimensions(self) -> "Image":
        if self.width is None or self.height is None:
            self.width = self.size[0]
            self.height = self.size[1]
        return self

    def to_row(self) -> dict:
        return {f"image_{key}": value for key, value in self.model_dump().items()}

    @classmethod
    def from_row(cls, row: dict) -> "Image":
        return cls(
            **{
                k.replace("image_", ""): v
                for k, v in row.items()
                if k.startswith("image_")
            }
        )

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
        if isinstance(value, str):
            value = _base64_to_image(value)
        elif isinstance(value, np.ndarray):
            try:
                value = PIL.Image.fromarray(value)
            except Exception:
                value = to_tensor(value)
        return value

    @field_serializer("content")
    def serialize_content(self, content: torch.Tensor | PILImage | None, _info) -> str:
        """
        Serializes the image tensor to a base64-encoded string.

        Args:
            content (Optional[Union[torch.Tensor, PILImage]]): The image tensor to serialize.
            _info: Additional information (unused).

        Returns:
            str: The serialized image as a base64-encoded string.
        """
        if content is None:
            return None
        return _image_to_base64(content)

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_shapes(cls, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            assert value.ndim in [
                2,
                3,
            ], (
                f"Invalid number of dimensions in the image tensor: {value.shape}. Image tensor must be 2D (grayscale) or 3D (channels, height, width)."
            )
            if value.ndim == 2:
                value = value.unsqueeze(0)
        return value

    def load_content(self):
        """
        Loads the image content from the file path or URL.
        If the `file_path` is a URL, it fetches the image data using requests.
        If the `file_path` is a local file, it opens the image using PIL.
        Raises:
            ValueError: If the image content is not loaded.
            FileNotFoundError: If the image file does not exist.
        """
        assert self.file_path is not None, "Image file path is not set."
        if str(self.file_path).startswith(("http", "https")):
            import requests

            response = requests.get(self.file_path)
            if response.status_code != 200:
                raise ValueError(f"Failed to load image from URL: {self.file_path}")
            return _bytes_to_image(response.content)
        else:
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Image file not found: {self.file_path}")
            return PILImageModule.open(self.file_path)

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
            if self.content is None:
                raise ValueError(
                    "Image content is not loaded. Call load_content() first."
                )
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
            if isinstance(self.content, torch.Tensor):
                self.content = to_pil_image(self.content)
            self._is_tensor = False
        return self

    def convert_to_rgb(self) -> "Image":
        """
        Converts the image content to RGB format.

        Raises:
            ValueError: If the image content is not loaded.
        """
        if self.content is None:
            raise ValueError("Image content is not loaded.")
        if self.channels == 1:
            if isinstance(self.content, PILImage):
                self.content = self.content.convert("RGB")
            elif isinstance(self.content, torch.Tensor):
                self.content = self.content.repeat(3, 1, 1)
        return self

    def resize(self, width: int, height: int) -> "Image":
        """
        Resizes the image to the specified size.

        Args:
            width (int): The desired width of the image.
            height (int): The desired height of the image.

        Returns:
            Image: The resized image.
        """
        if self.content is None:
            raise ValueError("Image content is not loaded.")
        if isinstance(self.content, PILImage):
            from PIL.Image import Resampling

            self.content = self.content.resize(
                (width, height), resample=Resampling.BICUBIC
            )
        elif isinstance(self.content, torch.Tensor):
            from torchvision.transforms.functional import InterpolationMode, resize

            self.content = resize(
                self.content, (height, width), interpolation=InterpolationMode.BILINEAR
            )
        return self

    def normalize(
        self,
        mean: float | tuple[float, ...],
        std: float | tuple[float, ...],
    ) -> "Image":
        """
        Normalizes the image tensor using the specified mean and standard deviation.

        Args:
            mean (Union[float, Tuple[float, ...]]): The mean value(s) for normalization.
            std (Union[float, Tuple[float, ...]]): The standard deviation value(s) for normalization.

        Returns:
            Image: The normalized image.
        """
        if self.content is None:
            raise ValueError("Image content is not loaded.")
        if isinstance(self.content, torch.Tensor):
            from torchvision.transforms.functional import normalize

            self.content = normalize(self.content, mean=mean, std=std)
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
        import imagesize

        if self.file_path is not None:
            return imagesize.get(self.file_path)
        if isinstance(self.content, torch.Tensor):
            return (self.content.shape[2], self.content.shape[1])
        elif isinstance(self.content, PILImage):
            return self.content.size
        else:
            raise ValueError("Image content is not loaded or has an unsupported type.")

    @property
    def channels(self) -> int:
        """
        Returns the number of channels in the image.

        Returns:
            int: The number of channels in the image.
        """
        if isinstance(self.content, torch.Tensor):
            return self.content.shape[0]
        elif isinstance(self.content, PILImage):
            return len(self.content.getbands())
