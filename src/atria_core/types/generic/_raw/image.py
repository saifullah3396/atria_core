from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import model_validator

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import RawDataModel
from atria_core.types.typing.common import OptIntField, OptStrField
from atria_core.utilities.encoding import ValidatedPILImage

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.image import TensorImage  # noqa


logger = get_logger(__name__)


class Image(RawDataModel["TensorImage"]):
    _tensor_model = "atria_core.types.generic._tensor.image.TensorImage"
    file_path: OptStrField = None
    content: ValidatedPILImage = None
    width: OptIntField | None = None
    height: OptIntField | None = None

    @model_validator(mode="after")
    def _validate_dims(self):
        if self.width is None or self.height is None:
            if self.content is not None:
                self._set_skip_validation("width", self.content.size[0])
                self._set_skip_validation("height", self.content.size[1])
            elif self.file_path is not None and Path(self.file_path).exists():
                import imagesize

                size = imagesize.get(self.file_path)
                self._set_skip_validation("width", size[0])
                self._set_skip_validation("height", size[1])
        return self

    @property
    def size(self) -> tuple[int, int] | None:
        return (
            (self.width, self.height)
            if self.width is not None and self.height is not None
            else None
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        assert self.content is not None, "Image content is not loaded."
        return (len(self.content.getbands()), *self.content.size)

    @property
    def channels(self) -> int:
        assert self.content is not None, "Image content is not loaded."
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

    def to_rgb(self) -> "Image":
        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        self.content = self.content.convert("RGB")
        return self

    def to_grayscale(self) -> "Image":
        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        self.content = self.content.convert("L")
        return self

    def resize(self, width: int, height: int) -> "Image":
        assert self.content is not None, (
            "Image content is not loaded. Call load() first."
        )
        from PIL.Image import Resampling

        self.content = self.content.resize((width, height), resample=Resampling.BICUBIC)
        self.width, self.height = width, height
        return self
