import ast
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import field_serializer, field_validator

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.common import OCRType
from atria_core.types.typing.common import OptStrField, TableSchemaMetadata
from atria_core.utilities.file import _load_bytes_from_uri


class OCR(BaseDataModel):
    _batch_merge_fields: ClassVar[list[str] | None] = ["type"]
    file_path: OptStrField = None
    type: Annotated[OCRType | None, TableSchemaMetadata(pa_type="string")] = None
    content: Annotated[str | None, TableSchemaMetadata(pa_type="binary")] = None

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, value: Any) -> str | None:
        if isinstance(value, Path):
            return str(value)
        return value

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, value: Any) -> str | None:
        if isinstance(value, str):
            return OCRType(value)
        return value

    def _load(self):
        if self.content is None:
            if self.file_path is None:
                raise ValueError("Either file_path or content must be provided.")

            self.content = _load_bytes_from_uri(self.file_path)
            if self.content.startswith("b'"):
                self.content = ast.literal_eval(self.content).decode("utf-8")

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
