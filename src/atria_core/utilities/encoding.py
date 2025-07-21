"""
Encoding Utilities Module

This module provides utility functions for encoding and decoding images, strings,
and other data formats. It includes functionality for converting images to bytes
or base64 strings, compressing and decompressing strings, and handling various
data transformations.

Functions:
    - _pil_image_to_bytes: Converts a PIL image to a byte array.
    - _image_to_bytes: Converts an image (PIL, Tensor, or ndarray) to a byte array.
    - _image_to_base64: Converts an image to a base64-encoded string.
    - _bytes_to_image: Converts a byte array to a PIL image.
    - _base64_to_image: Converts a base64-encoded string to a PIL image.
    - _compress_string: Compresses a string using gzip.
    - _encode_string: Encodes a string into a base64-encoded compressed format.
    - _decompress_string: Decompresses a gzip-compressed byte string.
    - _decode_string: Decodes a base64-encoded compressed string.

Dependencies:
    - base64: For base64 encoding and decoding.
    - gzip: For compressing and decompressing strings.
    - io: For handling in-memory byte streams.
    - numpy: For handling image data in ndarray format.
    - PIL: For image processing.
    - torch: For handling image data in Tensor format.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING, Union

from PIL.Image import Image as PILImage

if TYPE_CHECKING:
    import numpy as np
    import PIL
    import PIL.Image
    import torch


def _pil_image_to_bytes(image: "PILImage", format: str = "PNG") -> bytes:
    """
    Converts a PIL image to a byte array.

    Args:
        image (PILImage): The PIL image to convert.
        format (str): The format to save the image in (e.g., "PNG"). Defaults to "PNG".

    Returns:
        bytes: The byte array representation of the image.
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def _image_to_bytes(
    image: Union["PILImage", "torch.Tensor", "np.ndarray"], format: str = "PNG"
) -> bytes:
    """
    Converts an image (PIL, Tensor, or ndarray) to a byte array.

    Args:
        image (Union[PILImage, torch.Tensor, np.ndarray]): The image to convert.
        format (str): The format to save the image in (e.g., "PNG"). Defaults to "PNG".

    Returns:
        bytes: The byte array representation of the image.

    Raises:
        TypeError: If the image type is unsupported.
    """
    from PIL.Image import Image as PILImage

    if isinstance(image, PILImage):
        return _pil_image_to_bytes(image, format=format)
    else:
        import numpy as np
        import torch
        from torchvision.transforms.functional import to_pil_image

        if isinstance(image, torch.Tensor | np.ndarray):
            return _pil_image_to_bytes(to_pil_image(image), format=format)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


def _image_to_base64(image: Union["PILImage", "torch.Tensor", "np.ndarray"]) -> str:
    """
    Converts an image to a base64-encoded string.

    Args:
        image (Union[PILImage, torch.Tensor, np.ndarray]): The image to encode.

    Returns:
        str: The base64-encoded string representation of the image.
    """
    import base64

    return base64.b64encode(_image_to_bytes(image)).decode("utf-8")


def _bytes_to_image(encoded_image: bytes) -> "PILImage":
    """
    Converts a byte array to a PIL image.

    Args:
        encoded_image (bytes): The byte array representation of the image.

    Returns:
        PILImage: The decoded PIL image.
    """
    import io

    from PIL import Image

    return Image.open(io.BytesIO(encoded_image))


def _base64_to_image(encoded_image: str) -> "PILImage":
    """
    Converts a base64-encoded string to a PIL image.

    Args:
        encoded_image (str): The base64-encoded string representation of the image.

    Returns:
        PILImage: The decoded PIL image.
    """
    import base64
    import io

    return PIL.Image.open(io.BytesIO(base64.b64decode(encoded_image)))


def _compress_string(input: str) -> bytes:
    """
    Compresses a string using gzip.

    Args:
        input (str): The string to compress.

    Returns:
        bytes: The compressed byte string.
    """
    import gzip
    import io

    with io.BytesIO() as buffer:
        with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
            f.write(input.encode("utf-8"))
        return buffer.getvalue()


def _encode_string(input: str) -> str:
    """
    Encodes a string into a base64-encoded compressed format.

    Args:
        input (str): The string to encode.

    Returns:
        str: The base64-encoded compressed string.
    """
    import base64

    return base64.b64encode(_compress_string(input)).decode("utf-8")


def _decompress_string(input: bytes) -> str:
    """
    Decompresses a gzip-compressed byte string.

    Args:
        input (bytes): The compressed byte string to decompress.

    Returns:
        str: The decompressed string.
    """
    import gzip
    import io

    with io.BytesIO(input) as buffer:
        with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
            return f.read().decode("utf-8")


def _decode_string(input: str | bytes) -> str:
    """
    Decodes a base64-encoded compressed string.

    Args:
        input (Union[str, bytes]): The encoded string or byte string to decode.

    Returns:
        str: The decoded string.

    Raises:
        gzip.BadGzipFile: If the input is not a valid gzip-compressed string.
    """
    import base64
    import gzip

    if isinstance(input, bytes):
        try:
            return _decompress_string(base64.b64decode(input))
        except gzip.BadGzipFile:
            return input.decode("utf-8")
    return input
