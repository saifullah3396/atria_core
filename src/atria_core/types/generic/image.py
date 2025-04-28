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
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image as PILImage
from pydantic import computed_field, field_serializer, field_validator, model_validator

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel, BatchedBaseDataModel
from atria_core.types.typing.common import PydanticFilePath
from atria_core.utilities.encoding import _base64_to_image, _image_to_base64

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
            value = _base64_to_image(value)
        elif isinstance(value, np.ndarray):
            try:
                value = PIL.Image.fromarray(value)
            except Exception as e:
                value = to_tensor(value)
        return value

    @field_serializer("content")
    def serialize_content(
        self, content: Optional[Union[torch.Tensor, PILImage]], _info
    ) -> str:
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

        if not self._is_tensor:
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

        if self._is_tensor:
            logger.debug(f"Converting {self.__class__.__name__} from tensors.")
            if isinstance(self.content, torch.Tensor):
                self.content = to_pil_image(self.content)
            self._is_tensor = False
        return self

    @computed_field
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

    @computed_field
    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the image tensor.

        Returns:
            torch.dtype: The data type of the image tensor.
        """
        if isinstance(self.content, torch.Tensor):
            return self.content.dtype

    @field_serializer("dtype")
    def serialize_dtype(self, dtype: torch.dtype, _info) -> str:
        """
        Serializes the data type of the image tensor to a string.

        Args:
            dtype (torch.dtype): The data type of the image tensor.
            _info: Additional information (unused).

        Returns:
            str: The serialized data type as a string.
        """
        return str(dtype)

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

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedImage":
        """
        Constructs a new BatchedImage instance using the provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments used to initialize the BatchedImage.

        Returns:
            BatchedImage: A new instance of BatchedImage.
        """
        return BatchedImage(**kwargs)


class BatchedImage(BatchedBaseDataModel):
    """
    A class for handling batched image data.

    This class accommodates both a list of individual image tensors and a single tensor
    representing a batch of images.

    Attributes:
        file_path (List[PydanticFilePath]): A list of file paths associated with the images.
        content (Union[torch.Tensor, List[Image], List[torch.Tensor]]): The image data, which must be either:
            - A list of tensors with shape (C, H, W), or
            - A single tensor with shape (N, C, H, W), where N represents the batch size.
        source_size (List[Tuple[int,int]]): A list representing the source dimensions of the images.
    """

    file_path: List[PydanticFilePath] | None
    content: Union[torch.Tensor, List[PILImage], List[torch.Tensor]]
    source_size: List[Optional[Tuple[int, int]]] | None

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_shapes(cls, value: Any) -> torch.Tensor:
        """
        Validates the "content" field after processing.

        Args:
            value (Any): The input value for the "content" field.

        Returns:
            torch.Tensor: The validated tensor.

        Raises:
            AssertionError: If the tensor dimensions are invalid.
        """
        if isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                assert all(
                    x.ndim == 3 for x in value
                ), "content must either be a list of tensors of shape (C, H, W) or a single tensor of shape (N, C, H, W)"
        else:
            if isinstance(value, torch.Tensor):
                assert (
                    value.ndim == 4
                ), "content must either be a list of tensors of shape (C, H, W) or a single tensor of shape (N, C, H, W)"
        return value

    @computed_field
    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the image tensor.

        Returns:
            torch.Size: The shape of the image tensor.
        """
        self._validate_is_tensor()
        if isinstance(self.content, list):
            return [x.shape for x in self.content]
        return self.content.shape

    @property
    def size(self) -> torch.Size:
        """
        Returns the size of the image as (width, height).

        Returns:
            torch.Size: The size of the image.
        """
        self._validate_is_tensor()
        shape = self.content.shape
        if isinstance(shape, list):
            return [(s[3], s[2]) for s in shape]
        return (shape[3], shape[2])

    @property
    def width(self) -> int:
        """
        Returns the width of the image.

        Returns:
            int: The width of the image.
        """
        self._validate_is_tensor()
        size = self.size
        if isinstance(size, list):
            return [s[0] for s in size]
        return size[0]

    @property
    def height(self) -> int:
        """
        Returns the height of the image.

        Returns:
            int: The height of the image.
        """
        self._validate_is_tensor()
        size = self.size
        if isinstance(size, list):
            return [s[1] for s in size]
        return size[1]

    @computed_field
    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the image tensor.

        Returns:
            torch.dtype: The data type of the image tensor.
        """
        self._validate_is_tensor()
        if isinstance(self.content, list):
            return [x.dtype for x in self.content]
        return self.content.dtype
