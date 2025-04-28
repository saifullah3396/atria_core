"""
Tokenized Object Data Instance Module

This module defines the `TokenizedObjectInstance` and `BatchedTokenizedObjectInstance` classes, which represent data instances
containing tokenized information. These classes are designed to handle tokenized data such as token IDs, word IDs, token labels,
and other related attributes. They are useful for tasks like token classification, sequence labeling, and object detection in
tokenized formats.

Classes:
    - TokenizedObjectInstance: Represents a single tokenized data instance.
    - BatchedTokenizedObjectInstance: Represents a batch of tokenized data instances.

Dependencies:
    - pydantic: For data validation and serialization.
    - atria_core.data_types.data_instance.base: For the base data instance class.
    - atria_core.data_types.generic.image: For handling image data.
    - atria_core.data_types.generic.label: For handling label data.
    - atria_core.data_types.typing.tensor: For defining tensor types.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import torch
from pydantic import model_validator

from atria_core.types.data_instance.base import (
    BaseDataInstance,
    BatchedBaseDataInstance,
)
from atria_core.types.generic.image import BatchedImage, Image
from atria_core.types.generic.label import BatchedLabel, Label


class TokenizedObjectInstance(BaseDataInstance):
    """
    Represents a single tokenized data instance.

    This class is designed to handle tokenized data such as token IDs, word IDs,
    token labels, and other related attributes. It is useful for tasks like token
    classification, sequence labeling, and object detection in tokenized formats.

    Attributes:
        token_ids (PydanticTensor): The token IDs for the instance.
        word_ids (torch.Tensor | None): The word IDs corresponding to the tokens. Defaults to None.
        sequence_ids (torch.Tensor | None): The sequence IDs for the tokens. Defaults to None.
        token_labels (torch.Tensor | None): The labels for each token. Defaults to None.
        token_type_ids (torch.Tensor | None): The token type IDs for the instance. Defaults to None.
        attention_mask (torch.Tensor | None): The attention mask for the tokens. Defaults to None.
        token_bboxes (torch.Tensor | None): The bounding boxes for tokens (if applicable). Defaults to None.
        label (Label | None): The label associated with the instance. Defaults to None.
        image (Image | None): The image associated with the instance (if applicable). Defaults to None.
    """

    token_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    word_ids: torch.Tensor | None = None
    sequence_ids: torch.Tensor | None = None
    image: Image | None = None
    label: Label | None = None

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedTokenizedObjectInstance":
        """
        Constructs a batch of TokenizedObjectInstance objects from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments containing lists of attributes.

        Returns:
            BatchedTokenizedObjectInstance: A BatchedTokenizedObjectInstance object containing lists of attributes.
        """
        return BatchedTokenizedObjectInstance(**kwargs)

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "TokenizedObjectInstance":
        """
        Validates the shapes of the tensors in the instance.

        This method checks that the shapes of the token IDs, word IDs, and other
        tensors are consistent with each other. It raises a ValueError if any
        inconsistencies are found.

        Returns:
            TokenizedObjectInstance: The validated instance.

        Raises:
            AssertionError: If the tensor shapes are inconsistent.
        """
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key in [
                "token_ids",
                "word_ids",
                "token_labels",
                "token_type_ids",
                "attention_mask",
            ]:
                assert (
                    value.ndim == 1
                ), f"Expected 1D tensor, but got {value.ndim}D tensor for {key}."
                assert (
                    value.shape == self.token_ids.shape
                ), f"{key} must have the same shape as token_ids {self.token_ids.shape}."
            if key == "token_bboxes":
                assert (
                    value.ndim == 2
                ), f"Expected 2D tensor, but got {value.ndim}D tensor for {key}."
                assert (
                    value.shape[0] == self.token_ids.shape[0] and value.shape[1] == 4
                ), f"{key} must have compatible shape with token_ids {self.token_ids.shape}."
        return self


class BatchedTokenizedObjectInstance(BatchedBaseDataInstance):
    """
    Represents a batch of tokenized data instances.

    This class is designed to handle batched tokenized data such as token IDs, word IDs,
    token labels, and other related attributes. It is useful for tasks like token
    classification, sequence labeling, and object detection in tokenized formats.

    Attributes:
        token_ids (PydanticTensor): The token IDs for the batch.
        word_ids (torch.Tensor | None): The word IDs corresponding to the tokens in the batch. Defaults to None.
        sequence_ids (torch.Tensor | None): The sequence IDs for the tokens in the batch. Defaults to None.
        token_labels (torch.Tensor | None): The labels for each token in the batch. Defaults to None.
        token_type_ids (torch.Tensor | None): The token type IDs for the batch. Defaults to None.
        attention_mask (torch.Tensor | None): The attention mask for the tokens in the batch. Defaults to None.
        token_bboxes (torch.Tensor | None): The bounding boxes for tokens in the batch (if applicable). Defaults to None.
        label (BatchedLabel | None): The labels associated with the batch. Defaults to None.
        image (BatchedImage | None): The images associated with the batch (if applicable). Defaults to None.
    """

    token_ids: torch.Tensor
    word_ids: torch.Tensor | None = None
    sequence_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    token_bboxes: torch.Tensor | None = None
    label: BatchedLabel | None = None
    image: BatchedImage | None = None

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "BatchedTokenizedObjectInstance":
        """
        Validates the shapes of the tensors in the instance.

        This method checks that the shapes of the token IDs, word IDs, and other
        tensors are consistent with each other. It raises a ValueError if any
        inconsistencies are found.

        Returns:
            BatchedTokenizedObjectInstance: The validated instance.

        Raises:
            AssertionError: If the tensor shapes are inconsistent.
        """
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key in [
                "token_ids",
                "word_ids",
                "token_labels",
                "token_type_ids",
                "attention_mask",
            ]:
                assert (
                    value.ndim == 2
                ), f"Expected 2D tensor, but got {value.ndim}D tensor for {key}."
                assert (
                    value.shape == self.token_ids.shape
                ), f"{key} must have the same shape as token_ids {self.token_ids.shape}."
            if key == "token_bboxes":
                assert (
                    value.ndim == 3
                ), f"Expected 3D tensor, but got {value.ndim}D tensor for {key}."
                assert (
                    value.shape[:2] == self.token_ids.shape[:2] and value.shape[-1] == 4
                ), f"{key} must have compatible shape with token_ids {self.token_ids.shape}."
        return self
