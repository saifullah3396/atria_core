import types
from functools import lru_cache
from types import NoneType
from typing import Any, Self, Union, get_args, get_origin, get_type_hints

import pyarrow as pa
from atria_core.logger.logger import get_logger
from atria_core.types.typing.common import TableSchemaMetadata
from pydantic import BaseModel

logger = get_logger(__name__)


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """
    Recursively flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested fields
        sep: Separator to use between nested keys

    Returns:
        dict[str, Any]: Flattened dictionary

    Example:
        >>> _flatten_dict({"a": {"b": 1, "c": 2}})
        {"a_b": 1, "a_c": 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(flat: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    result = {}

    for key, sub_schema in schema.items():
        if isinstance(sub_schema, dict):
            nested = {
                k[len(key) + 1 :]: v for k, v in flat.items() if k.startswith(f"{key}_")
            }
            if all(value is None for value in nested.values()):
                result[key] = None
            else:
                result[key] = _unflatten_dict(nested, sub_schema)
        else:
            # Primitive value
            if key in flat:
                result[key] = flat[key]

    return result


def _extract_pyarrow_schema(model_cls: type[BaseModel]) -> dict[str, type | dict]:
    """
    Extract PyArrow schema from a TableSerializable model class.

    Args:
        model_cls: Model class to extract schema from

    Returns:
        dict[str, Any]: Schema mapping field names to PyArrow types

    Raises:
        TypeError: If model_cls is not a valid TableSerializable class
    """
    if not issubclass(model_cls, BaseModel) or not issubclass(
        model_cls, TableSerializable
    ):
        raise TypeError(
            f"Expected a subclass of {TableSerializable} subclass, got {model_cls}"
        )

    schema: dict[str, type | dict] = {}
    import torch  # noqa: F401

    type_hints = get_type_hints(model_cls, include_extras=True)
    for field_name, annotated_type in type_hints.items():
        try:
            origin = get_origin(annotated_type)
            args = get_args(annotated_type)

            # Handle Optional[Annotated[...]] or Union types
            if origin in {Union, types.UnionType} and len(args) > 1:
                # Strip NoneType from Union to handle Optional
                non_none_args = [arg for arg in args if arg is not NoneType]
                if len(non_none_args) == 1:
                    annotated_type = non_none_args[0]
                    origin = get_origin(annotated_type)
                    args = get_args(annotated_type)

            # Handle Annotated types
            if hasattr(annotated_type, "__metadata__"):  # Annotated type
                base_type = args[0] if args else annotated_type
                metadata = (
                    args[1:]
                    if len(args) > 1
                    else getattr(annotated_type, "__metadata__", [])
                )

                # Check if base_type is a TableSerializable subclass
                if isinstance(base_type, type) and issubclass(
                    base_type, TableSerializable
                ):
                    nested_schema = _extract_pyarrow_schema(base_type)
                    # Flatten nested schema with prefixed field names
                    for nested_field_name, nested_field_type in nested_schema.items():
                        schema[f"{field_name}_{nested_field_name}"] = nested_field_type
                else:
                    # Search for TableSchemaMetadata in metadata
                    for meta in metadata:
                        if isinstance(meta, TableSchemaMetadata):
                            schema[field_name] = meta.pyarrow
                            break
                    else:
                        logger.debug(
                            f"No TableSchemaMetadata found for field {field_name}"
                        )

            # Handle direct TableSerializable subclass (not annotated)
            elif isinstance(annotated_type, type) and issubclass(
                annotated_type, TableSerializable
            ):
                nested_schema = _extract_pyarrow_schema(annotated_type)
                schema[field_name] = nested_schema

        except Exception as e:
            raise RuntimeError(
                f"Failed to process field {field_name} in {model_cls.__name__}"
            ) from e

    return schema


class TableSerializable(BaseModel):
    """
    A mixin class that provides table serialization capabilities for Pydantic models.

    This class enables conversion between Pydantic models and tabular data formats
    like PyArrow tables, with support for nested structures and custom field metadata.

    Features:
        - Automatic PyArrow schema generation from type annotations
        - Flattening/unflattening of nested structures
        - Row-wise serialization and deserialization
        - Caching for performance optimization

    Example:
        ```python
        from typing import Annotated


        class MyModel(TableSerializable):
            name: Annotated[str, TableSchemaMetadata(pa.string())]
            age: Annotated[int, TableSchemaMetadata(pa.int32())]


        model = MyModel(name="John", age=30)
        row = model.to_row()
        reconstructed = MyModel.from_row(row)
        ```
    """

    @classmethod
    @lru_cache(maxsize=128)
    def table_schema(cls) -> dict[str, Any]:
        """
        Get the table schema for this model class.

        Returns:
            dict[str, Any]: Schema mapping field names to PyArrow types or nested schemas

        Note:
            Results are cached for performance. Clear cache with cls.table_schema.cache_clear()
        """
        return _extract_pyarrow_schema(cls)

    @classmethod
    @lru_cache(maxsize=1)
    def table_schema_flattened(cls) -> dict[str, Any]:
        """
        Get the flattened table schema for this model class.

        Returns:
            dict[str, Any]: Flattened schema with nested field names joined by underscores
        """
        return _flatten_dict(cls.table_schema())

    @classmethod
    @lru_cache(maxsize=1)
    def pa_schema(cls) -> pa.Schema:
        """
        Get the PyArrow schema for this model class.

        Returns:
            pa.Schema: PyArrow schema object that can be used for table creation

        Raises:
            ValueError: If schema contains invalid PyArrow types
        """
        try:
            schema_items = list(cls.table_schema_flattened().items())
            return pa.schema(schema_items)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create PyArrow schema for {cls.__name__}"
            ) from e

    def to_row(self, include_none: bool = True) -> dict[str, Any]:
        """
        Convert this model instance to a flattened row dictionary.

        Args:
            include_none: Whether to include fields with None values in the output

        Returns:
            dict[str, Any]: Flattened row data suitable for table insertion

        Example:
            ```python
            model = MyModel(name="John", age=30)
            row = model.to_row()
            # {"name": "John", "age": 30}
            ```
        """
        try:
            schema = self.table_schema_flattened()
            data = _flatten_dict(self.model_dump())

            if include_none:
                return {k: data.get(k) for k in schema}
            else:
                return {k: v for k, v in data.items() if k in schema and v is not None}

        except Exception as e:
            raise RuntimeError(
                f"Failed to convert {self.__class__.__name__} to row"
            ) from e

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        """
        Create a model instance from a flattened row dictionary.

        Args:
            row: Flattened row data
            strict: If True, raise errors for missing fields. If False, use None for missing fields.

        Returns:
            Self: New model instance created from the row data

        Raises:
            ValueError: If row data is invalid or missing required fields (when strict=True)

        Example:
            ```python
            row = {"name": "John", "age": 30}
            model = MyModel.from_row(row)
            ```
        """
        try:
            return cls(**_unflatten_dict(row, cls.table_schema()))

        except Exception as e:
            raise RuntimeError(f"Failed to create {cls.__name__} from row") from e

    @classmethod
    def clear_schema_cache(cls) -> None:
        """Clear all cached schema data for this class."""
        cls.table_schema.cache_clear()
        cls.table_schema_flattened.cache_clear()
        cls.pa_schema.cache_clear()

    def get_table_fields(self) -> dict[str, Any]:
        """
        Get only the fields that are part of the table schema.

        Returns:
            dict[str, Any]: Dictionary containing only table-serializable fields
        """
        schema_fields = set(self.table_schema_flattened().keys())
        flattened_data = _flatten_dict(self.model_dump())
        return {k: v for k, v in flattened_data.items() if k in schema_fields}
