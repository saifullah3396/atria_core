from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel, PrivateAttr

from atria_core.logger.logger import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class Batchable(BaseModel):
    """
    A mixin class that provides batching functionality for Pydantic models.

    This class allows multiple instances of a model to be combined into a single
    batched instance, with intelligent handling of different field types including
    tensors, nested Batchable objects, and lists.

    Attributes:
        _is_batched: Private attribute indicating if this instance is a batched version.
        _batch_skip_fields: List of field names to skip during batching (set to None).
        _batch_merge_fields: List of field names to merge if values are consistent.
        _batch_tensor_stack_skip_fields: List of tensor fields to skip stacking.

    Example:
        ```python
        class MyModel(Batchable):
            data: torch.Tensor
            label: str


        instances = [MyModel(data=tensor1, label="a"), MyModel(data=tensor2, label="b")]
        batched = MyModel.batched(instances)
        ```
    """

    _batch_size: int | None = PrivateAttr(default=None)
    _is_batched: bool = PrivateAttr(default=False)
    _batch_skip_fields: ClassVar[list[str] | None] = None
    _batch_merge_fields: ClassVar[list[str] | None] = None
    _batch_tensor_stack_skip_fields: ClassVar[list[str] | None] = None

    @property
    def batch_size(self) -> int | None:
        """
        Get the batch size of this instance.

        Returns:
            int | None: The batch size if this instance is batched, None otherwise.
        """
        if not self.is_batched:
            raise ValueError("Attempted to access batch_size on a non-batched instance")
        return self._batch_size

    @property
    def is_batched(self) -> bool:
        """
        Check if this instance represents a batched collection of models.

        Returns:
            bool: True if this instance is batched, False otherwise.
        """
        return self._is_batched

    @classmethod
    def batched(cls, model_instances: list[Self]) -> Self:
        """
        Create a batched instance from a list of model instances.

        This method combines multiple instances of the same model class into a single
        batched instance. Different field types are handled intelligently:
        - Tensors are stacked if shapes match
        - Nested Batchable objects are recursively batched
        - Lists of Batchable objects are batched
        - Other values are collected into lists

        Args:
            model_instances: List of model instances to batch together.

        Returns:
            Self: A new batched instance containing the combined data.

        Raises:
            ValueError: If the input list is empty.

        Example:
            ```python
            instances = [model1, model2, model3]
            batched_model = MyModel.batched(instances)
            assert batched_model.is_batched == True
            ```
        """
        if not model_instances:
            raise ValueError("Cannot batch an empty list of model instances")

        batched_fields: dict[str, Any] = {}

        # Process each field defined in the model
        for field_name in cls.model_fields:
            try:
                # Extract values for this field from all instances
                values = [getattr(item, field_name) for item in model_instances]
                batched_fields[field_name] = cls._batch_field(field_name, values)
            except Exception as e:
                raise RuntimeError(f"Failed to batch field '{field_name}'") from e

        # Create the batched instance and mark it as batched
        batched_instance = cls.model_construct(**batched_fields)
        batched_instance._is_batched = True
        batched_instance._batch_size = len(model_instances)
        return batched_instance

    @classmethod
    def _batch_field(cls, field_name: str, values: list[Any]) -> Any:
        """
        Batch a specific field based on its type and configuration.

        This method determines how to combine values for a specific field based on:
        - Whether all values are None
        - Field-specific batching rules (skip, merge, tensor handling)
        - The type of the field values

        Args:
            field_name: Name of the field being batched.
            values: List of values for this field from all instances.

        Returns:
            Any: The batched value for this field.
        """
        import torch

        # If all values are None, return None
        if any(v is None for v in values):
            return None

        # Skip fields marked for skipping
        if cls._batch_skip_fields and field_name in cls._batch_skip_fields:
            return None

        # Handle merge fields (return single value if consistent, otherwise list)
        if cls._batch_merge_fields and field_name in cls._batch_merge_fields:
            return cls._handle_merge_field(field_name, values)

        first_value = values[0]

        # Handle nested Batchable objects
        if isinstance(first_value, Batchable):
            return cls._handle_nested_batchable(values)

        # Handle lists of Batchable objects
        if (
            isinstance(first_value, list)
            and first_value
            and isinstance(first_value[0], Batchable)
        ):
            return cls._handle_nested_list_of_batchables(values)

        # Handle torch tensors
        if isinstance(first_value, torch.Tensor):
            return cls._handle_tensor_field(field_name, values)

        # Fallback: return list of raw values
        return values

    @classmethod
    def _handle_merge_field(cls, field_name: str, values: list[Any]) -> Any:
        """
        Handle fields marked for merging by returning a single value if consistent.

        For fields in _batch_merge_fields, this method checks if all values are
        identical. If so, returns the single value. Otherwise, returns the full list.

        Args:
            field_name: Name of the field being processed.
            values: List of values to potentially merge.

        Returns:
            Any: Single value if all are identical, otherwise the full list.
        """
        if all(v == values[0] for v in values):
            return values[0]
        return values

    @classmethod
    def _handle_nested_batchable(cls, values: list["Batchable"]) -> "Batchable":
        """
        Handle batching of nested Batchable objects.

        When a field contains Batchable objects, this method recursively batches
        them using their own batching logic.

        Args:
            values: List of Batchable objects to batch.

        Returns:
            Self: A batched instance of the nested Batchable class, or the original
                list if batching fails.
        """
        return values[0].__class__.batched(values)

    @classmethod
    def _handle_nested_list_of_batchables(cls, values: list[list[Self]]) -> Any:
        """
        Handle batching of lists containing Batchable objects.

        When a field contains lists of Batchable objects, this method:
        1. Batches each sublist individually
        2. Then batches the resulting batched objects together

        Args:
            values: List of lists, where each inner list contains Batchable objects.

        Returns:
            Any: A batched instance containing the batched sublists, or the original
                list structure if batching fails.
        """
        try:
            nested_cls = values[0][0].__class__
            # First batch each sublist
            batched_list = [nested_cls.batched(sublist) for sublist in values]
            # Then batch the batched sublists
            return nested_cls.batched(batched_list)
        except Exception as e:
            raise RuntimeError(
                f"Failed to batch nested list of Batchables in field '{cls.__name__}': "
                f"Unable to batch sublists or combine batched results"
            ) from e

    @classmethod
    def _handle_tensor_field(cls, field_name: str, values: list["torch.Tensor"]) -> Any:
        """
        Handle batching of PyTorch tensor fields.

        For tensor fields, this method attempts to stack tensors along a new dimension
        if their shapes are compatible. Tensors in _batch_tensor_stack_skip_fields
        are returned as lists instead of being stacked.

        Args:
            field_name: Name of the tensor field being processed.
            values: List of PyTorch tensors to batch.

        Returns:
            Any: Stacked tensor if shapes match and stacking is not skipped,
                otherwise the original list of tensors.
        """
        import torch

        # Skip stacking for fields in the skip list
        if (
            cls._batch_tensor_stack_skip_fields
            and field_name in cls._batch_tensor_stack_skip_fields
        ):
            return values

        # Stack tensors if all shapes match
        if all(v.shape == values[0].shape for v in values):
            return torch.stack(values)
        return values
