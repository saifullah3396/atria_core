from functools import partial
from typing import TYPE_CHECKING, Generic, Self

from pydantic import BaseModel, PrivateAttr

from atria_core.logger.logger import get_logger
from atria_core.types.base._mixins._utils import _recursive_apply
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

    _device: "torch.device | str" = PrivateAttr(default="cpu")

    @property
    def device(self) -> "torch.device | str":
        """
        Get the current device where the model's tensors are stored.

        Returns:
            Optional[torch.device]: The device where the model's tensors are stored,
                                  or None if not initialized.
        """
        return self._device

    def to_device(self, device: "torch.device | str" = "cpu") -> Self:
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
        if self._device != device:
            from atria_core.utilities.tensors import _convert_to_device

            apply_results = _recursive_apply(
                self, ToDeviceConvertible, partial(_convert_to_device, device=device)
            )  # type: ignore[return-value]
            device_instance = self.model_validate(
                apply_results, context={"no_validation": True}
            )
            device_instance._device = device
            return device_instance
        return self

    def to_gpu(self, gpu_id: int = 0) -> Self:
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

    def to_cpu(self) -> Self:
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
