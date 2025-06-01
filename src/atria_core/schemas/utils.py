import hashlib
import json
import re
import uuid
from typing import Annotated

from pydantic import AfterValidator, BeforeValidator, PlainSerializer


def _convert_dict_to_schema(data: dict) -> dict:
    schema = {}
    for key, value in data.items():
        if isinstance(value, str):
            schema[key] = {"type": "string"}
        elif isinstance(value, int):
            schema[key] = {"type": "integer"}
        elif isinstance(value, float):
            schema[key] = {"type": "float"}
        elif isinstance(value, list):
            # Handle list elements, assuming a homogeneous list (list of same type)
            if len(value) > 0:
                # If the list contains dictionaries, process them recursively
                if isinstance(value[0], dict):
                    schema[key] = {
                        "type": "list",
                        "schema": _convert_dict_to_schema(value[0]),
                    }
                else:
                    schema[key] = {
                        "type": "list",
                        "schema": {"type": type(value[0]).__name__},
                    }
            else:
                schema[key] = {"type": "list", "schema": {}}
        elif isinstance(value, dict):
            # Recursively generate schema for nested dictionaries
            schema[key] = {"type": "dict", "schema": _convert_dict_to_schema(value)}
        else:
            schema[key] = {"type": "string"}  # Default type for unrecognized types
    return schema


def _generate_hash_from_dict(schema: dict) -> str:
    sorted_schema = json.dumps(schema, sort_keys=True)
    hash_object = hashlib.sha256(sorted_schema.encode())
    return hash_object.hexdigest()


def validate_name(input: str):
    """
    Validate name whether it is a valid string (allows alphanumeric, underscore, slash, hyphen, and dot).
    """
    if not isinstance(input, str):
        raise ValueError("Name must be a string")

    input = input.strip()

    if len(input) < 3 or len(input) > 100:
        raise ValueError(
            "Name must be at least 3 characters long and at most 100 characters"
        )

    pattern = r"^[a-zA-Z0-9_.\-/]+$"
    if not re.match(pattern, input):
        raise ValueError(
            "Name can only contain alphanumeric characters, underscores, slashes, hyphens, and dots"
        )
    return input


NameStr = Annotated[str, AfterValidator(validate_name)]
SerializableUUID = Annotated[
    uuid.UUID,
    BeforeValidator(lambda v: uuid.UUID(v) if not isinstance(v, uuid.UUID) else v),
    PlainSerializer(lambda x: str(x)),
]
SerializableDateTime = Annotated[
    str,
    BeforeValidator(lambda v: v.isoformat() if hasattr(v, "isoformat") else v),
    PlainSerializer(lambda x: x.isoformat() if hasattr(x, "isoformat") else x),
]
