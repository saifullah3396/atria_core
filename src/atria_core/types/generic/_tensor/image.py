from typing import TYPE_CHECKING, Any

import torch
from pydantic import computed_field, field_validator, model_validator

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import TensorDataModel

if TYPE_CHECKING:
    from atria_core.types.generic._raw.image import Image  # noqa

logger = get_logger(__name__)


class TensorImage(TensorDataModel["Image"]):
    _raw_model = "atria_core.types.generic._raw.image.Image"
    content: torch.Tensor

    def to_raw(self) -> "Image":
        from torchvision.transforms.functional import to_pil_image

        return self.raw_data_model()(content=to_pil_image(self.content))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def shape(self) -> "torch.Size":
        return self.content.shape

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dtype(self) -> "torch.dtype":
        return self.content.dtype

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size(self) -> "torch.Size":
        return torch.Size((self.content.shape[-1], self.content.shape[-2]))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def channels(self) -> int:
        if self._is_batched:
            return self.content.shape[1]
        else:
            return self.content.shape[0]

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content_before(cls, value: Any) -> Any | None:
        import numpy as np
        from PIL.Image import Image as PILImage
        from torchvision.transforms.functional import to_tensor

        if isinstance(value, PILImage | np.ndarray):
            return to_tensor(value)
        return value

    @model_validator(mode="after")  # type: ignore[misc]
    def _validate_content_after(self) -> Any | None:
        if isinstance(self.content, torch.Tensor):
            if self._is_batched:
                assert self.content.ndim == 4, (
                    f"Invalid number of dimensions in the image tensor: {self.content.shape}. "
                    f"Image tensor must be 4D (batch, channels, height, width)"
                )
            else:
                assert self.content.ndim in [2, 3], (
                    f"Invalid number of dimensions in the image tensor: {self.content.shape}. "
                    f"Image tensor must be 2D (grayscale) or 3D (channels, height, width)"
                )
            if self.content.ndim == 2:
                self.content = self.content.unsqueeze(0)

        return self

    def to_rgb(self) -> "TensorImage":
        repeats = (1, 3, 1, 1) if self._is_batched else (3, 1, 1)
        self.content = self.content.repeat(*repeats)
        return self

    def to_grayscale(self) -> "TensorImage":
        from torchvision.transforms.functional import rgb_to_grayscale

        self.content = rgb_to_grayscale(self.content, num_output_channels=1)
        return self

    def resize(self, width: int, height: int) -> "TensorImage":
        from torchvision.transforms.functional import InterpolationMode, resize

        self.content = resize(
            self.content, [height, width], interpolation=InterpolationMode.BICUBIC
        )
        return self

    def normalize(
        self, mean: float | tuple[float, ...], std: float | tuple[float, ...]
    ) -> "TensorImage":
        from torchvision.transforms.functional import normalize

        mean_list = list(mean) if isinstance(mean, tuple) else [mean]
        std_list = list(std) if isinstance(std, tuple) else [std]
        self.content = normalize(self.content, mean=mean_list, std=std_list)
        return self
