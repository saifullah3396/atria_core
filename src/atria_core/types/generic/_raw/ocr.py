import ast
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
from pydantic import field_serializer, field_validator

from atria_core.types.base.data_model import RawDataModel
from atria_core.types.enums import OCRType
from atria_core.types.typing.common import PydanticFilePath, TableSchemaMetadata

if TYPE_CHECKING:
    from atria_core.types.generic._tensor.ocr import TensorOCR  # noqa


class OCR(RawDataModel["TensorOCR"]):
    _tensor_model = "atria_core.types.generic._tensor.ocr.TensorOCR"
    file_path: PydanticFilePath = None
    type: Annotated[OCRType | None, TableSchemaMetadata(pyarrow=pa.string())] = None
    content: Annotated[str | None, TableSchemaMetadata(pyarrow=pa.binary())] = None

    def _load(self):
        if self.content is None:
            if self.file_path is None:
                raise ValueError("Either file_path or content must be provided.")
            if str(self.file_path).startswith(("http", "https")):
                import requests

                response = requests.get(self.file_path)
                if response.status_code != 200:
                    raise ValueError(f"Failed to load image from URL: {self.file_path}")
                self.content = response.content.decode("utf-8")
            else:
                if not Path(self.file_path).exists():
                    raise FileNotFoundError(f"File not found: {self.file_path}")
                with open(self.file_path, encoding="utf-8") as f:
                    self.content = f.read()
                    if self.content.startswith("b'"):
                        self.content = ast.literal_eval(self.content).decode("utf-8")
                    assert len(self.content) > 0, "OCR content is empty."

    def _unload(self):
        self.content = None

    @field_serializer("content")
    def _serialize_content(self, value: str | None) -> bytes | None:
        from atria_core.utilities.encoding import _compress_string

        if value is None:
            return None
        return _compress_string(value)

    @field_validator("content", mode="before")
    def _validate_content(cls, value: Any) -> str | None:
        from atria_core.utilities.encoding import _decompress_string

        if value is None:
            return None
        if isinstance(value, bytes):
            return _decompress_string(value)
        return value
