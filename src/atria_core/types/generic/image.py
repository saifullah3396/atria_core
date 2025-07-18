from pathlib import Path
from typing import TYPE_CHECKING, Self, Union

from pydantic import model_validator
from rich.repr import RichReprResult

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import OptIntField, OptStrField
from atria_core.utilities.encoding import ValidatedPILImage

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class Image(BaseDataModel):
    file_path: OptStrField = None
    content: ValidatedPILImage = None
    source_width: OptIntField = None
    source_height: OptIntField = None

    @property
    def width(self) -> int:
        return self.size[0] if self.size else None

    @property
    def height(self) -> int:
        return self.size[1] if self.size else None

    @model_validator(mode="after")
    def _validate_dims(self):
        if self.source_width is None or self.source_height is None:
            if self.content is not None:
                self._set_skip_validation("source_width", self.content.size[0])
                self._set_skip_validation("source_height", self.content.size[1])
            elif self.file_path is not None and Path(self.file_path).exists():
                import imagesize

                size = imagesize.get(self.file_path)
                self._set_skip_validation("source_width", size[0])
                self._set_skip_validation("source_height", size[1])
        return self

    @property
    def dtype(self) -> "torch.dtype":
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first or assign content directly."
        )
        if isinstance(self.content, torch.Tensor):
            import torch

            assert isinstance(self.content, torch.Tensor), (
                "Image content is not a tensor. Cannot get dtype."
            )
            return self.content.dtype
        else:
            raise ValueError("Image content is not a tensor. Cannot get dtype.")

    @property
    def size(self) -> tuple[int, int] | None:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first or assign content directly."
        )
        if isinstance(self.content, torch.Tensor):
            import torch

            assert isinstance(self.content, torch.Tensor), (
                "Image content is not a tensor. Cannot get dtype."
            )
            return (self.content.shape[-1], self.content.shape[-2])
        else:
            if self._is_batched:
                raise ValueError(
                    "Size is not defined for batched images. Use size instead."
                )
            return self.content.size

    @property
    def shape(self) -> Union[tuple[int, ...], "torch.Size"]:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first or assign content directly."
        )
        if isinstance(self.content, torch.Tensor):
            import torch

            assert isinstance(self.content, torch.Tensor), (
                "Image content is not a tensor. Cannot get shape."
            )
            return self.content.shape
        else:
            if self._is_batched:
                raise ValueError(
                    "Shape is not defined for batched images. Use size instead."
                )
            return (len(self.content.getbands()), *self.content.size)

    @property
    def channels(self) -> int | list[int]:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first or assign content directly."
        )
        if isinstance(self.content, torch.Tensor):
            import torch

            assert isinstance(self.content, torch.Tensor), (
                "Image content is not a tensor. Cannot get channels."
            )
            return self.content.shape[1] if self._is_batched else self.content.shape[0]
        else:
            if self._is_batched:
                assert isinstance(self.content, list) and len(self.content) > 0, (
                    "Expected a list of PIL Images for batched images."
                )
                return len(self.content[0].getbands())
            else:
                return len(self.content.getbands())

    def _load(self):
        if self.content is None:
            from PIL import Image as PILImageModule

            from atria_core.utilities.encoding import _bytes_to_image

            if self.file_path is None:
                raise ValueError(
                    "Image file path is not set. Please set the file_path before loading the image."
                )
            if self.file_path.startswith(("http", "https")):
                import requests

                response = requests.get(self.file_path)
                if response.status_code != 200:
                    raise ValueError(f"Failed to load image from URL: {self.file_path}")
                self.content = _bytes_to_image(response.content)
            else:
                if not Path(self.file_path).exists():
                    raise FileNotFoundError(f"Image file not found: {self.file_path}")
                if not Path(self.file_path).is_file():
                    raise ValueError(
                        f"Provided file path is not a file: {self.file_path}"
                    )
                self.content = PILImageModule.open(self.file_path)

    def _unload(self) -> None:
        self.content = None

    def _to_tensor(self) -> None:
        import torch
        from torchvision.transforms.functional import to_tensor

        if self.content is not None and not isinstance(self.content, torch.Tensor):
            if self._is_batched:
                assert isinstance(self.content, list) and len(self.content) > 0, (
                    "Expected a list of PIL Images for batched images."
                )
                try:
                    self.content = torch.stack([to_tensor(img) for img in self.content])
                except Exception:
                    self.content = [to_tensor(img) for img in self.content]
            else:
                import PIL

                assert isinstance(self.content, PIL.Image.Image), (
                    "Image content is not a PIL Image. Cannot convert to tensor."
                )
                self.content = to_tensor(self.content)

    def _to_raw(self) -> None:
        import torch
        from torchvision.transforms.functional import to_pil_image

        if self.content is not None and isinstance(self.content, torch.Tensor):
            self.content = to_pil_image(self.content)

    def to_rgb(self) -> Self:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        if isinstance(self.content, torch.Tensor):
            repeats = (1, 3, 1, 1) if self._is_batched else (3, 1, 1)
            self.content = self.content.repeat(*repeats)
        else:
            self.content = (
                [x.convert("RGB") for x in self.content]
                if self._is_batched
                else self.content.convert("RGB")
            )
        return self

    def to_grayscale(self) -> Self:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        if isinstance(self.content, torch.Tensor):
            from torchvision.transforms.functional import rgb_to_grayscale

            self.content = rgb_to_grayscale(self.content, num_output_channels=1)
        else:
            self.content = (
                [x.convert("L") for x in self.content]
                if self._is_batched
                else self.content.convert("L")
            )
        return self

    def resize(self, width: int, height: int) -> Self:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        if isinstance(self.content, torch.Tensor):
            from torchvision.transforms.functional import InterpolationMode, resize

            self.content = resize(
                self.content, [height, width], interpolation=InterpolationMode.BICUBIC
            )
        else:
            from PIL.Image import Resampling

            if self._is_batched:
                assert (self.content, list) and len(self.content) > 0, (
                    "Expected a list of PIL Images for batched images."
                )
                self.content = [
                    x.resize((width, height), resample=Resampling.BICUBIC)
                    for x in self.content
                ]
                self.source_width, self.source_height = width, height
            else:
                self.content = self.content.resize(
                    (width, height), resample=Resampling.BICUBIC
                )
                self.source_width, self.source_height = width, height
        return self

    def normalize(
        self, mean: float | tuple[float, ...], std: float | tuple[float, ...]
    ) -> Self:
        import torch

        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        assert isinstance(self.content, torch.Tensor), (
            "Normalization is only supported for tensor images. "
            "Convert the image to tensor first using `to_tensor()`."
        )
        from torchvision.transforms.functional import normalize

        mean_list = list(mean) if isinstance(mean, tuple) else [mean]
        std_list = list(std) if isinstance(std, tuple) else [std]
        self.content = normalize(self.content, mean=mean_list, std=std_list)
        return self

    def __rich_repr__(self) -> RichReprResult:  # type: ignore
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        yield from super().__rich_repr__()
        if self.content is not None:
            yield "width", self.width
            yield "height", self.height
            yield "channels", self.channels
