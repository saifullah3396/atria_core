"""
Base Data Model Module

This module defines the `BaseDataModel` class, which provides a foundational structure for
data models in the system. It supports batching, device management, and recursive loading
of nested data models. The class is built on top of Pydantic for data validation and serialization.

Classes:
    - BaseDataModel: A base class for data models with batching, device management, and loading support.

Dependencies:
    - typing: For type annotations.
    - torch: For tensor operations and device management.
    - pydantic: For data validation and serialization.
    - rich.pretty: For pretty-printing representations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING, ClassVar, TypeVar

from atria_core.constants import _MAX_REPR_PRINT_ELEMENTS
from atria_core.logger.logger import get_logger
from atria_core.utilities.tensors import (
    _convert_from_tensor,
    _convert_to_tensor,
    _stack_tensors_if_possible,
)
from pydantic import BaseModel, ConfigDict, PrivateAttr
from rich.pretty import pretty_repr

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseDataModel")


def ungroup_by_repeats(flat_list, counts):
    grouped = []
    idx = 0
    for count in counts:
        grouped.append(flat_list[idx : idx + count])
        idx += count
    return grouped


class BaseDataModelConfigDict(ConfigDict):
    batch_skip_fields: list[str] = []
    batch_merge_fields: list[str] = []
    row_serialization_types: dict[str, type] = {}


class RowSerializable:
    _row_name: ClassVar[str | None] = None
    _row_serialization_types: ClassVar[dict[str, str]]

    def __init__(self) -> None:
        assert hasattr(self, "_row_serialization_types"), (
            f"{self.__class__.__name__} must define _row_serialization_types that maps row names to primitive types."
        )

    @classmethod
    def row_serialization_types(cls) -> dict[str, type]:
        """
        Returns the serialization types for the row representation of the model.

        This should be defined in subclasses to map row names to primitive types.
        """
        if cls._row_name is not None:
            return {
                f"{cls._row_name}_{key}": value
                for key, value in cls._row_serialization_types.items()
            }
        else:
            return cls._row_serialization_types

    def to_row(self) -> dict:
        if self._row_name is not None:
            return {
                f"{self._row_name}_{key}": value
                for key, value in self.model_dump().items()
            }
        else:
            return self.model_dump()

    @classmethod
    def from_row(cls, row: dict) -> "BaseDataModel":
        if cls._row_name is not None:
            return cls(
                **{
                    k.replace(f"{cls._row_name}_", ""): v
                    for k, v in row.items()
                    if k.startswith(cls._row_name)
                }
            )
        else:
            return cls(**row)


class BaseDataModel(BaseModel):
    """
    A base class for data models with batching, device management, and loading support.

    This class provides functionality for batching multiple instances of the model,
    managing device placement for tensors, and recursively loading nested data models.

    Attributes:
        model_config (ConfigDict): Configuration for the Pydantic model.
        _device (torch.device): The device where the model's tensors are stored.
    """

    model_config = BaseDataModelConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="forbid",
    )
    _device = PrivateAttr(default=None)
    _is_tensor = PrivateAttr(default=None)
    _is_batched = PrivateAttr(default=False)

    def model_post_init(self, context: ConfigDict) -> None:
        """
        Post-initialization method to set the device to CPU if not already set.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        import torch

        self._device = torch.device("cpu")

    @classmethod
    def batched(cls, model_instances: list[T]) -> T:
        """
        Recursively converts a list of data models into a single batched model.

        For each field, if its value is an instance of `BaseDataModel`, batch it recursively.
        Otherwise, collect the field values into a list.

        Args:
            model_instances (List[T]): A list of model instances to batch.

        Returns:
            T: A single batched instance of the model.
        """
        import torch

        batched_fields = {}
        for field_name, _ in cls.model_fields.items():
            # Gather all the values for this field from the list of models.
            values = [getattr(item, field_name) for item in model_instances]
            # print(field_name, values)

            # Check if all values are None, if so set None
            if all(value is None for value in values):
                batched_fields[field_name] = None
                continue

            # if this field name is in batch_ignored_fields, skip it
            if (
                "batch_skip_fields" in cls.model_config
                and field_name in cls.model_config["batch_skip_fields"]
            ):
                batched_fields[field_name] = None
                continue

            # if this field name is in batch_merge_fields, skip it
            if (
                "batch_merge_fields" in cls.model_config
                and field_name in cls.model_config["batch_merge_fields"]
            ):
                if all(v == values[0] for v in values):
                    batched_fields[field_name] = values[0]
                else:
                    logger.debug(
                        f"Field {field_name} has different values in the batch. Skipping merge for this field."
                    )
                    batched_fields[field_name] = values
                continue

            # If this is a list of lists of BaseDataModel, process inner lists first
            if (
                isinstance(values[0], list)
                and values[0]
                and isinstance(values[0][0], BaseDataModel)
            ):
                nested_cls = values[0][0].__class__
                batched_list_of_objects = []
                for list_of_objects in values:
                    batched_list_of_objects.append(nested_cls.batched(list_of_objects))
                batched_fields[field_name] = nested_cls.batched(batched_list_of_objects)
            elif isinstance(values[0], BaseDataModel):
                nested_cls = values[0].__class__
                batched_fields[field_name] = nested_cls.batched(values)
            elif isinstance(values[0], torch.Tensor):
                try:
                    if (
                        "batch_tensor_stack_skip_fields" in cls.model_config
                        and field_name
                        in cls.model_config["batch_tensor_stack_skip_fields"]
                    ):
                        batched_fields[field_name] = values
                    else:
                        if all(v.shape == values[0].shape for v in values):
                            batched_fields[field_name] = _stack_tensors_if_possible(
                                values
                            )
                        else:
                            batched_fields[field_name] = values
                except Exception as e:
                    logger.debug(f"Failed to stack tensors for field {field_name}: {e}")
                    batched_fields[field_name] = values
            else:
                batched_fields[field_name] = values

        # Create a new instance of the model with the batched fields and with no validation
        # if list of instances is empty, we reach here directly
        batched_instance = cls.model_construct(**batched_fields)
        batched_instance._is_tensor = all(
            instance._is_tensor for instance in model_instances
        )
        batched_instance._is_batched = True
        return batched_instance

    @property
    def device(self) -> "torch.device":
        """
        Returns the current device where the model's tensors are stored.

        If no tensor fields are present, returns CPU.

        Returns:
            torch.device: The device where the model's tensors are stored.
        """
        return self._device

    def _validate_is_tensor(self: T) -> T:
        assert self._is_tensor, (
            f"This operation is only supported for tensor-based {self.__class__.__name__}. Call to_tensor() first."
        )

    def to_tensor(self: T) -> T:
        """
        Converts the model's data to tensors.

        This method is intended for converting the model's data into tensor format.

        Returns:
            T: The model instance with data converted to tensors.
        """
        if not self._is_tensor or self._is_tensor is None:
            for field_name, field_value in self.__dict__.items():
                if isinstance(field_value, BaseDataModel):
                    setattr(self, field_name, field_value.to_tensor())
                elif isinstance(field_value, list):
                    if len(field_value) == 0:
                        continue
                    if isinstance(field_value[0], BaseDataModel):
                        new_list = [item.to_tensor() for item in field_value]
                    else:
                        new_list = _convert_to_tensor(field_value)
                    setattr(self, field_name, new_list)
                else:
                    setattr(self, field_name, _convert_to_tensor(field_value))
            self._is_tensor = True
        return self

    def from_tensor(self: T) -> T:
        """
        Converts the model's data from tensors.

        This method is intended for converting the model's data back from tensor format.

        Returns:
            T: The model instance with data converted from tensors.
        """
        if self._is_tensor or self._is_tensor is None:
            for field_name, field_value in self.__dict__.items():
                if isinstance(field_value, BaseDataModel):
                    setattr(self, field_name, field_value.from_tensor())
                elif isinstance(field_value, list):
                    new_list = []
                    for item in field_value:
                        if isinstance(item, BaseDataModel):
                            new_list.append(item.from_tensor())
                        else:
                            new_list.append(_convert_from_tensor(item))
                    setattr(self, field_name, new_list)
                else:
                    setattr(self, field_name, _convert_from_tensor(field_value))
            self._is_tensor = False
        return self

    def model_dump(self, *args, **kwargs):
        self.from_tensor()
        return super().model_dump(*args, round_trip=True, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        self.from_tensor()
        return super().model_dump_json(*args, round_trip=True, **kwargs)

    def to_device(self: T, device: "torch.device" = "cpu") -> T:
        """
        Moves the model's tensors to the specified device.

        Args:
            device (torch.device): The target device.

        Returns:
            T: The model instance with tensors moved to the target device.

        Raises:
            AssertionError: If the model data is not loaded as tensor.
        """
        import torch

        assert self._is_tensor, "Data is not loaded as tensor. Call to_tensor() first."
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, torch.Tensor):
                setattr(self, field_name, field_value.to(device))
            elif isinstance(field_value, BaseDataModel):
                setattr(self, field_name, field_value.to_device(device))
            elif isinstance(field_value, list):
                new_list = []
                for item in field_value:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.to(device))
                    elif isinstance(item, BaseDataModel):
                        new_list.append(item.to_device(device))
                    else:
                        new_list.append(item)
                setattr(self, field_name, new_list)

        self._device = device
        return self

    def to_gpu(self: T) -> T:
        """
        Moves the model's tensors to the GPU.

        Returns:
            T: The model instance with tensors moved to the GPU.
        """
        import ignite.distributed as idist

        self.to_device(idist.device())
        return self

    def to_cpu(self: T) -> T:
        """
        Moves the model's tensors to the CPU.

        Returns:
            T: The model instance with tensors moved to the CPU.
        """
        import torch

        self.to_device(torch.device("cpu"))
        return self

    def repeat_with_indices(
        self: T, repeat_indices: list[int], ignored_fields: list[str]
    ) -> T:
        import torch

        assert self._is_tensor, (
            "This function only supports tensorized inputs. Call to_tensor() first."
        )
        assert self._is_batched, (
            "This function only supports batched inputs. Call batched() on a list of instances first."
        )
        if not hasattr(self, "_is_repeated"):
            for field_name, field_value in self.__dict__.items():
                try:
                    if field_name in ["_repeat_indices"]:
                        continue
                    if field_name in ignored_fields:
                        continue
                    if isinstance(field_value, BaseDataModel):
                        setattr(
                            self,
                            field_name,
                            field_value.repeat_with_indices(
                                repeat_indices, ignored_fields
                            ),
                        )
                    elif isinstance(field_value, list):
                        if len(field_value) == 0:
                            continue
                        setattr(
                            self,
                            field_name,
                            [
                                item
                                for item, count in zip(field_value, repeat_indices)
                                for _ in range(count)
                            ],
                        )
                    elif isinstance(field_value, torch.Tensor):
                        setattr(
                            self,
                            field_name,
                            field_value.repeat_interleave(
                                torch.tensor(repeat_indices, device=self.device), dim=0
                            ),
                        )
                    else:
                        setattr(self, field_name, field_value)
                except Exception as e:
                    logger.error(
                        f"Failed to repeat field {field_name} with indices {repeat_indices}: {e}"
                    )
                    raise e
            setattr(self, "_is_repeated", True)
        return self

    def gather_with_indices(
        self: T, gather_indices: list[int], ignored_fields: list[str]
    ) -> T:
        import torch

        assert self._is_tensor, "Only supports tensorized inputs"
        assert self._is_batched, "Only supports batched inputs"
        assert hasattr(self, "_is_repeated"), "Input does not appear to be repeated"
        for field_name, field_value in self.__dict__.items():
            if field_name in [
                "_is_repeated",
            ]:
                continue

            if isinstance(field_value, BaseDataModel):
                setattr(
                    self,
                    field_name,
                    field_value.gather_with_indices(gather_indices, ignored_fields),
                )

            elif isinstance(field_value, list):
                if len(field_value) == 0:
                    continue
                if field_name not in ignored_fields:
                    # Just pick first item from each group
                    grouped = ungroup_by_repeats(field_value, gather_indices)
                    values = [group[0] for group in grouped]
                else:
                    values = ungroup_by_repeats(field_value, gather_indices)
                setattr(self, field_name, values)

            elif isinstance(field_value, torch.Tensor):
                split_tensors = torch.split(field_value, gather_indices, dim=0)
                if field_name not in ignored_fields:
                    values = torch.stack([tensor[0] for tensor in split_tensors])
                else:
                    values = list(split_tensors)
                setattr(self, field_name, values)
            else:
                setattr(self, field_name, field_value)

        delattr(self, "_is_repeated")
        return self

    def __repr__(self) -> str:
        """
        Returns a pretty-printed string representation of the model.

        Returns:
            str: The string representation of the model.
        """
        return pretty_repr(self, max_length=_MAX_REPR_PRINT_ELEMENTS)

    def __str__(self) -> str:
        """
        Returns a string representation of the model.

        Returns:
            str: The string representation of the model.
        """
        return pretty_repr(self, max_length=_MAX_REPR_PRINT_ELEMENTS)
