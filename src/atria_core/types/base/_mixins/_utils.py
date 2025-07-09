from collections.abc import Callable
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _recursive_apply_in_place(
    model_instance: T, model_type: type[T], apply_fn: Callable[[Any], Any]
) -> None:
    for field_name in model_instance.__class__.model_fields:
        try:
            value = getattr(model_instance, field_name)
            if isinstance(value, model_type):
                # Recurse into nested model
                _recursive_apply(value, model_type, apply_fn)
            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], model_type)
            ):
                raise RuntimeError(
                    f"Field '{field_name}' contains list of {model_type}, which is not supported."
                )
            else:
                apply_fn(value)

        except Exception as e:
            raise RuntimeError(f"Error converting field '{field_name}'") from e


def _recursive_apply(
    model_instance: T,
    model_type: type[T],
    apply_fn: Callable[[Any], Any],
    exclude_fields: set[str] | None = None,
) -> dict[str, Any]:
    apply_results: dict[str, Any] = {}

    for field_name in model_instance.__class__.model_fields:
        try:
            value = getattr(model_instance, field_name)
            if isinstance(value, model_type):
                # Recurse into nested model
                if exclude_fields is not None and field_name in exclude_fields:
                    apply_results[field_name] = value
                else:
                    # Apply the function recursively
                    apply_results[field_name] = _recursive_apply(
                        value, model_type, apply_fn, exclude_fields=exclude_fields
                    )
            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], model_type)
            ):
                raise RuntimeError(
                    f"Field '{field_name}' contains list of {model_type}, which is not supported."
                )
            else:
                if exclude_fields is not None and field_name in exclude_fields:
                    apply_results[field_name] = value
                else:
                    apply_results[field_name] = apply_fn(value)

        except Exception as e:
            raise RuntimeError(
                f"Error applying function to field '{field_name}'"
            ) from e
    return apply_results


def _contains_tensor_type(tp) -> bool:
    """
    Recursively checks if a given type contains any tensor type.

    Returns:
        True if any tensor type is found, otherwise False.
    """
    import torch

    origin = get_origin(tp)

    if origin is Union:
        # Check all arguments inside Union (ignore NoneType)
        return any(
            _contains_tensor_type(arg) for arg in get_args(tp) if arg is not type(None)
        )

    elif origin is list:
        # Check list element type
        return _contains_tensor_type(get_args(tp)[0])

    elif origin is Annotated:
        # Recursively check the wrapped type inside Annotated
        return _contains_tensor_type(get_args(tp)[0])

    elif tp in [torch.Tensor]:
        return True  # Found a tensor type

    return False  # No tensor types found
