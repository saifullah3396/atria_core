from typing import TYPE_CHECKING, Any, Self

from pydantic import PrivateAttr

from atria_core.logger.logger import get_logger
from atria_core.types.base._mixins._batchable import Batchable

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def _ungroup_by_repeats(flat_list: list[Any], counts: list[int]) -> list[list[Any]]:
    """
    Split a flat list into groups based on repeat counts.

    Args:
        flat_list: The flattened list to split
        counts: List of counts indicating group sizes

    Returns:
        list[list[Any]]: List of grouped items

    Raises:
        ValueError: If counts don't match the flat_list length

    Example:
        >>> _ungroup_by_repeats([1, 2, 3, 4, 5], [2, 3])
        [[1, 2], [3, 4, 5]]
    """
    if sum(counts) != len(flat_list):
        raise ValueError(
            f"Sum of counts ({sum(counts)}) doesn't match list length ({len(flat_list)})"
        )

    grouped = []
    idx = 0
    for count in counts:
        if count < 0:
            raise ValueError(f"Count must be non-negative, got {count}")
        grouped.append(flat_list[idx : idx + count])
        idx += count
    return grouped


class Repeatable(Batchable):
    """
    A mixin class that provides repeat and gather functionality for batched models.

    This class extends Batchable to support operations where elements need to be
    repeated or gathered based on indices. Useful for scenarios like beam search
    or variable-length sequence processing.

    Attributes:
        _is_repeated: Private attribute indicating if this instance has been repeated.

    Example:
        ```python
        class MyModel(Repeatable):
            data: torch.Tensor
            labels: list[str]


        # Create batched instances
        instances = [
            MyModel(data=tensor1, labels=["a"]),
            MyModel(data=tensor2, labels=["b"]),
        ]
        batched = MyModel.batched(instances)

        # Repeat elements
        repeated = batched.repeat_with_indices([2, 1], ignored_fields=[])

        # Gather back
        gathered = repeated.gather_with_indices([2, 1], ignored_fields=[])
        ```
    """

    _is_repeated: bool = PrivateAttr(default=False)

    @property
    def is_repeated(self) -> bool:
        """
        Check if this instance has been repeated.

        Returns:
            bool: True if this instance has been repeated, False otherwise.
        """
        return self._is_repeated

    def repeat_with_indices(
        self, repeat_indices: list[int], ignored_fields: list[str] | None = None
    ) -> Self:
        """
        Repeat elements in the batch according to the given indices.

        For each element in the batch, repeat it the number of times specified
        in repeat_indices. Different field types are handled appropriately:
        - Tensors: Use repeat_interleave
        - Lists: Repeat items using list comprehension
        - Nested Repeatable objects: Recursively apply repeat

        Args:
            repeat_indices: List of integers specifying how many times to repeat each element
            ignored_fields: List of field names to skip during repeat operation

        Returns:
            Self: This instance with repeated elements

        Raises:
            AssertionError: If this instance is not batched
            ValueError: If repeat_indices length doesn't match batch size
            RuntimeError: If field repetition fails

        Example:
            ```python
            # Batch has 2 elements, repeat first 3 times, second 1 time
            repeated = batched_model.repeat_with_indices([3, 1])
            # Result will have 4 elements total
            ```
        """
        if not self.is_batched:
            raise ValueError(
                "This function only supports batched inputs. Call batched() on a list of instances first."
            )

        if ignored_fields is None:
            ignored_fields = []

        if len(repeat_indices) != self.batch_size:
            raise ValueError(
                f"Length of repeat_indices ({len(repeat_indices)}) must match batch size ({self.batch_size})"
            )

        if any(idx < 0 for idx in repeat_indices):
            raise ValueError("All repeat indices must be non-negative")

        if self._is_repeated:
            return self

        repeated_values = {}
        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if field_name in ignored_fields:
                    repeated_values[field_name] = field_value
                repeated_values[field_name] = self._repeat_field(
                    field_value, repeat_indices, ignored_fields
                )
            except Exception as e:
                logger.error(f"Failed to repeat field '{field_name}': {e}")
                raise RuntimeError(
                    f"Failed to repeat field '{field_name}' with indices {repeat_indices}"
                ) from e

        self._is_repeated = True
        self._batch_size = sum(repeat_indices)
        return self.model_construct(**repeated_values)

    def _repeat_field(
        self, field_value: Any, repeat_indices: list[int], ignored_fields: list[str]
    ) -> Any:
        """
        Repeat a single field value according to repeat indices.

        Args:
            field_value: The field value to repeat
            repeat_indices: List of repeat counts
            ignored_fields: Fields to skip during recursion

        Returns:
            Any: The repeated field value
        """
        if field_value is None:
            return None

        if isinstance(field_value, Repeatable):
            return field_value.repeat_with_indices(repeat_indices, ignored_fields)

        elif isinstance(field_value, list):
            if len(field_value) == 0:
                return field_value
            if len(field_value) != len(repeat_indices):
                raise ValueError(
                    f"List length ({len(field_value)}) doesn't match repeat_indices length ({len(repeat_indices)})"
                )
            return [
                item
                for item, count in zip(field_value, repeat_indices, strict=True)
                for _ in range(count)
            ]

        elif isinstance(field_value, torch.Tensor):
            if field_value.size(0) != len(repeat_indices):
                raise ValueError(
                    f"Tensor batch size ({field_value.size(0)}) doesn't match repeat_indices length ({len(repeat_indices)})"
                )
            return field_value.repeat_interleave(
                torch.tensor(repeat_indices, device=field_value.device), dim=0
            )

        else:
            # For other types, return as-is
            return field_value

    def gather_with_indices(
        self, gather_indices: list[int], ignored_fields: list[str] | None = None
    ) -> Self:
        """
        Gather elements back from a repeated batch.

        This operation reverses the repeat operation by grouping repeated elements
        back together. For non-ignored fields, only the first element from each
        group is kept. For ignored fields, all elements are preserved as lists.

        Args:
            gather_indices: List of integers specifying the size of each group
            ignored_fields: List of field names to preserve all elements for

        Returns:
            Self: This instance with gathered elements

        Raises:
            AssertionError: If this instance is not batched
            ValueError: If gather_indices don't match current batch structure
            RuntimeError: If field gathering fails

        Example:
            ```python
            # If batch was repeated with [3, 1], gather back with [3, 1]
            gathered = repeated_model.gather_with_indices([3, 1])
            # Result will have 2 elements (original batch size)
            ```
        """
        if not self.is_batched:
            raise ValueError(
                "This function only supports batched inputs. Call batched() on a list of instances first."
            )

        if ignored_fields is None:
            ignored_fields = []

        if not self._is_repeated:
            logger.warning("Instance is not repeated. Skipping gather operation.")
            return self

        if sum(gather_indices) != self.batch_size:
            raise ValueError(
                f"Sum of gather_indices ({sum(gather_indices)}) must match current batch size ({self.batch_size})"
            )

        # Use model_fields instead of __dict__ for proper field iteration
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)

            try:
                new_value = self._gather_field(
                    field_value, gather_indices, ignored_fields, field_name
                )
                setattr(self, field_name, new_value)
            except Exception as e:
                logger.error(f"Failed to gather field '{field_name}': {e}")
                raise RuntimeError(
                    f"Failed to gather field '{field_name}' with indices {gather_indices}"
                ) from e

        self._is_repeated = False
        self._batch_size = len(gather_indices)
        return self

    def _gather_field(
        self,
        field_value: Any,
        gather_indices: list[int],
        ignored_fields: list[str],
        field_name: str,
    ) -> Any:
        """
        Gather a single field value according to gather indices.

        Args:
            field_value: The field value to gather
            gather_indices: List of group sizes
            ignored_fields: Fields to preserve all elements for
            field_name: Name of the field being processed

        Returns:
            Any: The gathered field value
        """
        if field_value is None:
            return None

        if isinstance(field_value, Repeatable):
            return field_value.gather_with_indices(gather_indices, ignored_fields)

        elif isinstance(field_value, list):
            if len(field_value) == 0:
                return field_value

            grouped = _ungroup_by_repeats(field_value, gather_indices)

            if field_name in ignored_fields:
                return grouped  # Return all elements as nested lists
            else:
                # Return first element from each group
                return [group[0] if group else None for group in grouped]

        elif isinstance(field_value, torch.Tensor):
            if field_value.size(0) != sum(gather_indices):
                raise ValueError(
                    f"Tensor size ({field_value.size(0)}) doesn't match sum of gather_indices ({sum(gather_indices)})"
                )

            split_tensors = torch.split(field_value, gather_indices, dim=0)

            if field_name in ignored_fields:
                return list(split_tensors)  # Return all tensors as list
            else:
                # Stack first elements from each group
                first_elements = [
                    tensor[0] if tensor.size(0) > 0 else torch.empty_like(tensor[0:1])
                    for tensor in split_tensors
                ]
                return torch.stack(first_elements)

        else:
            # For other types, return as-is
            return field_value

    def reset_repeat_state(self) -> Self:
        """
        Reset the repeat state without changing the data.

        Returns:
            Self: This instance with repeat state reset
        """
        self._is_repeated = False
        return self
