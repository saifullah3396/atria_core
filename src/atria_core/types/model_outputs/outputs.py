"""
Model Outputs Module

This module defines various model output classes that encapsulate the results of machine learning models.
These classes are designed to handle outputs for different tasks, including classification, token classification,
question answering, autoencoding, and diffusion models.

Classes:
    - AtriaModelOutput: Base class for all model outputs.
    - ClassificationModelOutput: Output for classification tasks.
    - TokenClassificationModelOutput: Output for token classification tasks.
    - LayoutTokenClassificationModelOutput: Output for layout token classification tasks.
    - QAModelOutput: Output for question answering tasks.
    - SequenceQAModelOutput: Output for sequence-based question answering tasks.
    - AutoEncoderModelOutput: Output for autoencoder models.
    - VarAutoEncoderModelOutput: Output for variational autoencoder models.
    - VarAutoEncoderGANModelOutput: Output for variational autoencoder GAN models.
    - DiffusionModelOutput: Output for diffusion models.
    - MMDetTrainingOutput: Output for MMDetection training tasks.
    - MMDetEvaluationOutput: Output for MMDetection evaluation tasks.

Dependencies:
    - torch: For tensor operations.
    - pydantic: For data validation and model definition.
    - diffusers: For handling variational autoencoder distributions.
    - mmdet: For handling MMDetection data samples.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from mmdet.structures import DetDataSample


@dataclass(frozen=True)
class ModelOutput:
    """
    Base class for all model outputs.

    Attributes:
        model_config (ConfigDict): Configuration for the model output.
        loss (Optional[""torch.Tensor""]): The loss value associated with the model output.
    """

    loss: torch.Tensor


@dataclass(frozen=True)
class ClassificationModelOutput(ModelOutput):
    """
    Output for classification tasks.

    Attributes:
        logits (Optional[""torch.Tensor""]): The raw logits from the model.
        prediction (Optional[""torch.Tensor""]): The predicted class labels.
        label (Optional[""torch.Tensor""]): The ground truth labels.
    """

    logits: torch.Tensor | None = None
    gt_label: torch.Tensor | None = None
    gt_label_name: list[str] | None = None
    predicted_label: torch.Tensor | None = None
    predicted_label_name: list[str] | None = None


@dataclass(frozen=True)
class TokenClassificationModelOutput(ModelOutput):
    """
    Output for token classification tasks.

    Attributes:
        logits (Optional[""torch.Tensor""]): The raw logits from the model.
        predicted_labels (Optional[List[List[str]]): The predicted labels for each token.
        target_labels (Optional[List[List[str]]): The ground truth labels for each token.
    """

    logits: torch.Tensor | None = None
    predicted_labels: list[list[str]] | None = None
    target_labels: list[list[str]] | None = None


@dataclass(frozen=True)
class LayoutTokenClassificationModelOutput(ModelOutput):
    """
    Output for layout token classification tasks.

    Attributes:
        logits (Optional[""torch.Tensor""]): The raw logits from the model.
        token_labels (Optional[""torch.Tensor""]): The ground truth token labels.
        token_bboxes (Optional[List[""torch.Tensor""]]): The bounding boxes for tokens.
        predicted_labels (Optional[List[List[str]]): The predicted labels for each token.
    """

    logits: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    token_bboxes: torch.Tensor | None = None
    predicted_labels: list[list[str]] | None = None


@dataclass(frozen=True)
class QAModelOutput(ModelOutput):
    """
    Output for question answering tasks.

    Attributes:
        pred_answers (Optional[List[str]]): The predicted answers.
        target_answers (Optional[List[str]]): The ground truth answers.
    """

    pred_answers: list[str] | None = None
    target_answers: list[str] | None = None


@dataclass(frozen=True)
class SequenceQAModelOutput(ModelOutput):
    """
    Output for sequence-based question answering tasks.

    Attributes:
        start_logits (Optional[""torch.Tensor""]): The logits for the start positions.
        end_logits (Optional[""torch.Tensor""]): The logits for the end positions.
        predicted_answers (Optional[List[str]]): The predicted answers.
        words (Optional[List[str]]): The words in the input sequence.
        word_ids (Optional[List[int]]): The word IDs in the input sequence.
        sequence_ids (Optional[List[int]]): The sequence IDs in the input sequence.
        question_id (Optional[int]): The ID of the question.
        gold_answers (Optional[List[str]]): The gold standard answers.
    """

    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    predicted_answers: list[str] | None = None
    words: list[list[str]] | None = None
    word_ids: torch.Tensor = None
    sequence_ids: torch.Tensor = None
    question_id: torch.Tensor = None
    gold_answers: list[list[str]] | None = None


@dataclass(frozen=True)
class AutoEncoderModelOutput(ModelOutput):
    """
    Output for autoencoder models.

    Attributes:
        real (Optional[""torch.Tensor""]): The original input tensor.
        reconstructed (Optional[""torch.Tensor""]): The reconstructed tensor.
    """

    real: torch.Tensor | None = None
    reconstructed: torch.Tensor | None = None


@dataclass(frozen=True)
class VarAutoEncoderModelOutput(ModelOutput):
    """
    Output for variational autoencoder models.

    Attributes:
        real (Optional[""torch.Tensor""]): The original input tensor.
        reconstructed (Optional[""torch.Tensor""]): The reconstructed tensor.
        posterior (Optional[DiagonalGaussianDistribution]): The posterior distribution.
        kl_loss (Optional[""torch.Tensor""]): The KL divergence loss.
        rec_loss (Optional[""torch.Tensor""]): The reconstruction loss.
    """

    real: torch.Tensor | None = None
    reconstructed: torch.Tensor | None = None
    posterior: DiagonalGaussianDistribution | None = None
    kl_loss: torch.Tensor | None = None
    rec_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class VarAutoEncoderGANModelOutput(ModelOutput):
    """
    Output for variational autoencoder GAN models.

    Attributes:
        real (Optional[""torch.Tensor""]): The original input tensor.
        reconstructed (Optional[""torch.Tensor""]): The reconstructed tensor.
        generated (Optional[""torch.Tensor""]): The generated tensor.
        kl_loss (Optional[""torch.Tensor""]): The KL divergence loss.
        nll_loss (Optional[""torch.Tensor""]): The negative log-likelihood loss.
        rec_loss (Optional[""torch.Tensor""]): The reconstruction loss.
        d_weight (Optional[""torch.Tensor""]): The discriminator weight.
        disc_factor (Optional[""torch.Tensor""]): The discriminator factor.
        g_loss (Optional[""torch.Tensor""]): The generator loss.
    """

    real: torch.Tensor | None = None
    reconstructed: torch.Tensor | None = None
    generated: torch.Tensor | None = None
    kl_loss: torch.Tensor | None = None
    nll_loss: torch.Tensor | None = None
    rec_loss: torch.Tensor | None = None
    d_weight: torch.Tensor | None = None
    disc_factor: torch.Tensor | None = None
    g_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class DiffusionModelOutput(ModelOutput):
    """
    Output for diffusion models.

    Attributes:
        real (Optional[""torch.Tensor""]): The original input tensor.
        generated (Optional[""torch.Tensor""]): The generated tensor.
    """

    real: torch.Tensor | None = None
    generated: torch.Tensor | None = None


@dataclass(frozen=True)
class MMDetTrainingOutput(ModelOutput):
    """
    Output for MMDetection training tasks.

    This class is currently a placeholder for MMDetection training outputs.
    """


@dataclass(frozen=True)
class MMDetEvaluationOutput(ModelOutput):
    """
    Output for MMDetection evaluation tasks.

    Attributes:
        det_data_samples (Optional[List[DetDataSample]]): The detection data samples.
    """

    det_data_samples: list[DetDataSample] | None = None
