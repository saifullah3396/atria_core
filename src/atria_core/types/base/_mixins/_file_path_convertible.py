from pathlib import Path

from pydantic import BaseModel

from atria_core.logger.logger import get_logger

logger = get_logger(__name__)


class FilePathConvertible(BaseModel):
    def to_relative_file_paths(self, data_dir: str) -> None:
        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if isinstance(field_value, FilePathConvertible):
                    field_value.to_relative_file_paths(data_dir=data_dir)
                elif "file_path" in field_name and field_value is not None:
                    path_obj = Path(field_value)
                    if path_obj.is_absolute():
                        relative_path = path_obj.relative_to(data_dir)
                        self._set_skip_validation(field_name, str(relative_path))
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to relative path: {e}"
                ) from e
        return self

    def to_absolute_file_paths(self, data_dir: str) -> None:
        for field_name in self.__class__.model_fields:
            try:
                field_value = getattr(self, field_name)
                if isinstance(field_value, FilePathConvertible):
                    field_value.to_absolute_file_paths(data_dir=data_dir)
                elif "file_path" in field_name and field_value is not None:
                    path_obj = Path(field_value)
                    if not path_obj.is_absolute():
                        absolute_path = Path(data_dir) / path_obj
                        self._set_skip_validation(
                            field_name, str(absolute_path.resolve())
                        )
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to absolute path: {e}"
                ) from e
        return self
