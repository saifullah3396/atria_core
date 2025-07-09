from functools import partial
from typing import TYPE_CHECKING, Any, Self

from pydantic import PrivateAttr

from atria_core.logger.logger import get_logger
from atria_core.types.base._mixins._batchable import Batchable
from atria_core.types.base._mixins._utils import _recursive_apply

if TYPE_CHECKING:
    pass

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
    _exclude_fields: set[str] | None = PrivateAttr(default=None)
    _repeat_indices: list[int] | None = PrivateAttr(default=None)

    @property
    def is_repeated(self) -> bool:
        """
        Check if this instance has been repeated.

        Returns:
            bool: True if this instance has been repeated, False otherwise.
        """
        return self._is_repeated

    def repeat(
        self, repeat_indices: list[int], exclude_fields: set[str] | None = None
    ) -> Self:
        if self._is_repeated:
            return self

        if len(repeat_indices) != self.batch_size:
            raise ValueError(
                f"Length of repeat_indices ({len(repeat_indices)}) must match batch size ({self.batch_size})"
            )

        if any(idx < 0 for idx in repeat_indices):
            raise ValueError("All repeat indices must be non-negative")

        apply_results = _recursive_apply(
            self,
            Repeatable,
            partial(self._repeat_field, repeat_indices=repeat_indices),
            exclude_fields=exclude_fields,
        )  # type: ignore[return-value]

        repeated_instance = self.model_validate(
            **apply_results, context={"no_validation": True}
        )
        repeated_instance._is_repeated = True
        repeated_instance._repeat_indices = repeat_indices
        repeated_instance._exclude_fields = exclude_fields
        repeated_instance._is_batched = True
        repeated_instance._batch_size = sum(repeat_indices)
        return repeated_instance

    def _repeat_field(self, field_value: Any, repeat_indices: list[int]) -> Any:
        import torch

        if isinstance(field_value, list):
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

        return field_value

    def undo_repeat(self) -> Self:
        if not self._is_repeated:
            return self

        assert self._repeat_indices is not None, "Repeat indices must be set"
        apply_results = _recursive_apply(
            self,
            Repeatable,
            partial(self._undo_repeat_on_field, repeat_indices=self._repeat_indices),
            exclude_fields=self._exclude_fields,
        )
        instance = self.model_validate(**apply_results, context={"no_validation": True})
        instance._batch_size = len(self._repeat_indices) if self._repeat_indices else 0
        instance._is_batched = True
        return instance

    def _undo_repeat_on_field(self, field_value: Any, repeat_indices: list[int]) -> Any:
        import torch

        if isinstance(field_value, list):
            if len(field_value) == 0:
                return field_value
            grouped = _ungroup_by_repeats(field_value, repeat_indices)
            return [group[0] if group else None for group in grouped]
        elif isinstance(field_value, torch.Tensor):
            if field_value.size(0) != sum(repeat_indices):
                raise ValueError(
                    f"Tensor size ({field_value.size(0)}) doesn't match sum of repeat_indices ({sum(repeat_indices)})"
                )
            split_tensors = torch.split(field_value, repeat_indices, dim=0)

            # Stack first elements from each group
            first_elements = [
                tensor[0] if tensor.size(0) > 0 else torch.empty_like(tensor[0:1])
                for tensor in split_tensors
            ]
            return torch.stack(first_elements)

        return field_value
