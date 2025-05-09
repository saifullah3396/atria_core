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

from typing import Tuple
import torch
from pydantic import model_validator

from atria_core.types.base.data_model import BaseDataModelConfigDict
from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from atria_core.types.generic.image import Image
from atria_core.types.generic.label import Label


class TokenizedDocumentInstance(BaseDataInstance):
    """
    Represents a single tokenized data instance.

    This class is designed to handle tokenized data such as token IDs, word IDs,
    token labels, and other related attributes. It is useful for tasks like token
    classification, sequence labeling, and object detection in tokenized formats.

    Attributes:
        token_ids (PydanticTensor): The token IDs for the instance.
        token_bboxes (PydanticTensor, optional): The bounding boxes for the tokens.
        token_type_ids (PydanticTensor, optional): The type IDs for the tokens.
        token_labels (PydanticTensor, optional): The labels for the tokens.
        attention_mask (PydanticTensor, optional): The attention mask for the tokens.
        word_ids (PydanticTensor, optional): The word IDs for the tokens.
        sequence_ids (PydanticTensor, optional): The sequence IDs for the tokens.
        overflow_to_sample_mapping (PydanticTensor, optional): Mapping from overflowed
            tokens to sample indices.
    """

    model_config = BaseDataModelConfigDict(
        batch_skip_fields=["ocr", "page_id", "total_num_pages"],
    )

    token_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    word_ids: torch.Tensor | None = None
    sequence_ids: torch.Tensor | None = None
    overflow_to_sample_mapping: torch.Tensor | None = None
    prediction_indices_mask: torch.Tensor | None = None
    image: Image | None = None
    label: Label | None = None

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "TokenizedDocumentInstance":
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
                    value.ndim == 2
                ), f"Expected 1D tensor, but got {value.ndim}D tensor for {key}."
                assert (
                    value.shape == self.token_ids.shape
                ), f"{key} must have the same shape as token_ids {self.token_ids.shape}."
            if key == "token_bboxes":
                assert (
                    value.ndim == 3
                ), f"Expected 2D tensor, but got {value.ndim}D tensor for {key}."
                assert (
                    value.shape[1] == self.token_ids.shape[1] and value.shape[2] == 4
                ), f"{key} must have compatible shape with token_ids {self.token_ids.shape}."

        prediction_indices_mask = torch.zeros_like(self.token_ids, dtype=torch.bool)
        for idx, word_ids_per_sample in enumerate(self.word_ids):
            if self.token_labels is not None:
                prediction_indices = [
                    i
                    for i in range(len(word_ids_per_sample))
                    if word_ids_per_sample[i] != word_ids_per_sample[i - 1]
                    and self.token_labels[idx][i] != -100
                ]
            else:
                prediction_indices = [
                    i
                    for i in range(len(word_ids_per_sample))
                    if word_ids_per_sample[i] != word_ids_per_sample[i - 1]
                ]
            prediction_indices_mask[idx][prediction_indices] = True
        self.prediction_indices_mask = prediction_indices_mask

        return self

    def select_all_overflow_samples(self) -> Tuple[bool, list[int], list[str]]:
        """
        Concatenates all overflowed samples into a single tensor.
        This method is useful for handling overflowed tokens in tokenized data.
        It ensures that the overflowed tokens are properly mapped to their
        corresponding samples.

        If token_ids has a list of tensors with shapes: [(1, 512), (2, 512)],
        This function will concatenate them into a single tensor with shape (3, 512).

        Other elements such as 'image', 'id', etc will be repeated accordingly.
        If images is a list of tensors with shapes: [(1, 3, 224, 224), (1, 3, 224, 224)],
        This function will concatenate them into a single tensor with shape (3, 3, 224, 224) with second image
        repeated twice. This is done for all elments in data instance that are not in 'non_repeated_keys'.

        Args:
            None
        Returns:
            Tuple[bool, list[int], list[str]]: A tuple containing:
                - A boolean indicating whether the concatenation was successful.
                - A list of indices indicating the number of times each sample was repeated.
                - A list of keys that were not repeated.
        """
        assert (
            self._is_tensor
        ), f"This function only supports tensorized document instances. Call to_tensor() first."
        assert (
            self._is_batched
        ), "This function only supports batched document instances. Call batched() first."
        if isinstance(self.token_ids, list):
            repeat_indices = [sample.shape[0] for sample in self.token_ids]
            not_repeated_keys = [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "prediction_indices_mask",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
            ]

            # we concatenate the tensors for the keys that are already repeated
            for key in not_repeated_keys:
                setattr(self, key, torch.cat(getattr(self, key)))

            # we recursively repeat all the batched samples with given indices
            self.repeat_with_indices(repeat_indices, not_repeated_keys)
            return True, repeat_indices, not_repeated_keys
        else:
            # if it is not a list then it must already be a tensor
            assert isinstance(self.token_ids, torch.Tensor)
        return False, None, None

    def select_random_overflow_samples(self):
        """
        Unlike concat_all_overflow_samples, this function randomly selects one sample
        from each overflowed sample and concatenates them into a single tensor.
        """
        assert (
            self._is_tensor
        ), f"This function only supports tensorized document instances. Call to_tensor() first."
        assert (
            self._is_batched
        ), "This function only supports batched document instances. Call batched() first."
        if isinstance(self.token_ids, list):
            random_select_ids = [
                torch.randint(0, sample.shape[0], size=(1,)).item()
                for sample in self.token_ids
            ]
            for key in [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "prediction_indices_mask",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
            ]:
                if key in self.__dict__:
                    self.__dict__[key] = torch.stack(
                        [
                            sample[idx]
                            for sample, idx in zip(
                                self.__dict__[key],
                                random_select_ids,
                            )
                        ]
                    )
                assert self.__dict__[key].shape[0] == len(self.__dict__[key]), (
                    f"Shape mismatch for {key}. "
                    f"Expected {len(self.__dict__[key])}, "
                    f"but got {self.__dict__[key].shape[0]}."
                )
        else:
            for key in [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "prediction_indices_mask",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
            ]:
                if key in self.__dict__:
                    self.__dict__[key] = self.__dict__[key].squeeze()

    def select_first_overflow_samples(self):
        """
        Unlike concat_all_overflow_samples, this function randomly selects one sample
        from each overflowed sample and concatenates them into a single tensor.
        """
        assert (
            self._is_tensor
        ), f"This function only supports tensorized document instances. Call to_tensor() first."
        assert (
            self._is_batched
        ), "This function only supports batched document instances. Call batched() first."
        if isinstance(self.token_ids, list):
            for key in [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "prediction_indices_mask",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
            ]:
                if key in self.__dict__:
                    self.__dict__[key] = torch.stack(
                        [sample[0] for sample in self.__dict__[key]]
                    )
                assert self.__dict__[key].shape[0] == len(self.__dict__[key]), (
                    f"Shape mismatch for {key}. "
                    f"Expected {len(self.__dict__[key])}, "
                    f"but got {self.__dict__[key].shape[0]}."
                )
        else:
            for key in [
                "token_ids",
                "token_bboxes",
                "token_labels",
                "prediction_indices_mask",
                "attention_mask",
                "word_ids",
                "sequence_ids",
                "overflow_to_sample_mapping",
            ]:
                if key in self.__dict__:
                    self.__dict__[key] = self.__dict__[key].squeeze()
