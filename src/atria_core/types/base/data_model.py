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

from typing import TYPE_CHECKING, List, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr
from rich.pretty import pretty_repr

from atria_core.constants import _MAX_REPR_PRINT_ELEMENTS
from atria_core.logger.logger import get_logger
from atria_core.utilities.tensors import (
    _convert_from_tensor,
    _convert_to_tensor,
    _stack_tensors_if_possible,
)

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseDataModel")


class BaseDataModel(BaseModel):
    """
    A base class for data models with batching, device management, and loading support.

    This class provides functionality for batching multiple instances of the model,
    managing device placement for tensors, and recursively loading nested data models.

    Attributes:
        model_config (ConfigDict): Configuration for the Pydantic model.
        _device (torch.device): The device where the model's tensors are stored.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="ignore"
    )
    _device = PrivateAttr(default=None)
    _is_tensor = PrivateAttr(default=None)

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
    def batched(cls, model_instances: List[T]) -> T:
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

            # Check if all values are None, if so set None
            if all(value is None for value in values):
                batched_fields[field_name] = None
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
                batched_fields[field_name] = _stack_tensors_if_possible(values)
            else:
                batched_fields[field_name] = values

        # Create a new instance of the model with the batched fields and with no validation
        batched_instance = cls.model_construct(**batched_fields)
        batched_instance._is_tensor = True
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
        assert (
            self._is_tensor
        ), f"This operation is only supported for tensor-based {self.__class__.__name__}. Call to_tensor() first."

    def to_tensor(self: T) -> T:
        """
        Converts the model's data to tensors.

        This method is intended for converting the model's data into tensor format.

        Returns:
            T: The model instance with data converted to tensors.
        """
        if not self._is_tensor or self._is_tensor is None:
            logger.debug(f"Converting {self.__class__.__name__} to tensors.")
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
            logger.debug(f"Converting {self.__class__.__name__} from tensors.")
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
        logger.debug(f"Moving {self.__class__.__name__} to device: {device}.")

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
