from typing import TYPE_CHECKING, Any, Generic, Optional

from pydantic import BaseModel, PrivateAttr

from atria_core.logger.logger import get_logger
from atria_core.types.base.types import T_RawModel

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class ToDeviceConvertible(BaseModel, Generic[T_RawModel]):
    """
    A mixin class for converting PyTorch tensors within Pydantic models to different devices.

    This class provides functionality to move tensors and nested ToDeviceConvertible objects
    between different compute devices (CPU, GPU, etc.) while maintaining the model structure.

    Attributes:
        _device: Private attribute storing the current device of the model's tensors.

    Example:
        ```python
        class MyModel(ToDeviceConvertible):
            data: torch.Tensor
            labels: list[torch.Tensor]


        model = MyModel(data=tensor, labels=[tensor1, tensor2])
        model_on_gpu = model.to_device("cuda:0")
        model_on_cpu = model.to_cpu()
        ```
    """

    _device: Optional["torch.device"] = PrivateAttr(default=None)

    @property
    def device(self) -> Optional["torch.device"]:
        """
        Get the current device where the model's tensors are stored.

        Returns:
            Optional[torch.device]: The device where the model's tensors are stored,
                                  or None if not initialized.
        """
        return self._device

    def to_device(
        self, device: "torch.device | str" = "cpu"
    ) -> "ToDeviceConvertible[T_RawModel]":
        """
        Move the model's tensors to the specified device.

        This method recursively moves all torch.Tensor fields and nested ToDeviceConvertible
        objects to the target device. Lists containing tensors or convertible objects are
        also processed recursively.

        Args:
            device: The target device. Can be a torch.device object or device string
                   (e.g., "cpu", "cuda:0", "mps").

        Returns:
            ToDeviceConvertible[T_RawModel]: Self, with all tensors moved to the target device.

        Raises:
            RuntimeError: If device conversion fails for any tensor.

        Example:
            ```python
            model = MyModel(data=torch.randn(10))
            gpu_model = model.to_device("cuda:0")
            assert gpu_model.device.type == "cuda"
            ```
        """
        import torch

        try:
            target_device = torch.device(device)

            # Process all model fields
            for field_name in self.__class__.model_fields:
                field_value = getattr(self, field_name)
                setattr(
                    self,
                    field_name,
                    self._convert_field_to_device(field_value, target_device),
                )

            self._device = target_device
            return self

        except Exception as e:
            logger.error(f"Failed to move model to device {device}: {e}")
            raise RuntimeError(f"Device conversion failed: {e}") from e

    def _convert_field_to_device(self, field_value: Any, device: "torch.device") -> Any:
        """
        Convert a single field value to the target device.

        Args:
            field_value: The field value to convert.
            device: Target device.

        Returns:
            Any: The converted field value.
        """
        import torch

        if field_value is None:
            return None

        if isinstance(field_value, torch.Tensor):
            return field_value.to(device)

        if isinstance(field_value, ToDeviceConvertible):
            return field_value.to_device(device)

        if isinstance(field_value, list | tuple):
            converted_items = [
                self._convert_field_to_device(item, device) for item in field_value
            ]
            return type(field_value)(converted_items)

        if isinstance(field_value, dict):
            return {
                key: self._convert_field_to_device(value, device)
                for key, value in field_value.items()
            }

        # Return unchanged for non-tensor types
        return field_value

    def to_gpu(self, gpu_id: int = 0) -> "ToDeviceConvertible[T_RawModel]":
        """
        Move the model's tensors to GPU.

        Args:
            gpu_id: GPU device ID to use (default: 0).

        Returns:
            ToDeviceConvertible[T_RawModel]: Self, with tensors moved to GPU.

        Raises:
            RuntimeError: If CUDA is not available or device is invalid.

        Example:
            ```python
            model = MyModel(data=torch.randn(10))
            gpu_model = model.to_gpu(gpu_id=1)  # Move to cuda:1
            ```
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. To install atria-core with GPU support, use: uv add atria-core --extra torch-gpu"
            )

        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} not available. Only {torch.cuda.device_count()} GPUs found"
            )

        return self.to_device(f"cuda:{gpu_id}")

    def to_cpu(self) -> "ToDeviceConvertible[T_RawModel]":
        """
        Move the model's tensors to CPU.

        Returns:
            ToDeviceConvertible[T_RawModel]: Self, with tensors moved to CPU.

        Example:
            ```python
            gpu_model = model.to_gpu()
            cpu_model = gpu_model.to_cpu()
            assert cpu_model.device.type == "cpu"
            ```
        """
        return self.to_device("cpu")
