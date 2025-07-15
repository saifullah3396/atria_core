from atria_core.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel

logger = get_logger(__name__)


def _assert_values_equal(value1, value2, float_rtolerance=1e-05):
    """
    Compare two values.
        - For floats, uses math.isclose with the provided relative tolerance.
        - For lists, recursively compares each corresponding element.
        - For BaseModel instances, recurses into assert_models_equal.
        - For other types, compares with equality.
    """
    import math

    import numpy as np
    import torch
    from pydantic import BaseModel

    # Compare floats with tolerance.
    if isinstance(value1, float) and isinstance(value2, float):
        assert math.isclose(value1, value2, rel_tol=float_rtolerance), (
            f"Float values not close: {value1} vs {value2}"
        )
    # Recursively compare lists.
    elif isinstance(value1, list) and isinstance(value2, list):
        assert len(value1) == len(value2), (
            f"List lengths don't match: {len(value1)} vs {len(value2)}"
        )
        for item1, item2 in zip(value1, value2, strict=True):
            _assert_values_equal(item1, item2, float_rtolerance)
    elif isinstance(value1, dict) and isinstance(value2, dict):
        assert value1.keys() == value2.keys(), (
            f"Dict keys differ: {value1.keys()} vs {value2.keys()}"
        )
        for key in value1:
            _assert_values_equal(value1[key], value2[key], float_rtolerance)
    elif isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        assert value1.shape == value2.shape, (
            f"Tensor shapes differ: {value1.shape} vs {value2.shape}"
        )
        assert value1.dtype == value2.dtype, (
            f"Tensor dtypes differ: {value1.dtype} vs {value2.dtype}"
        )
        assert torch.allclose(value1, value2, rtol=float_rtolerance), (
            f"Tensors not close: {value1} vs {value2}"
        )
    elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        assert value1.shape == value2.shape, (
            f"Tensor shapes differ: {value1.shape} vs {value2.shape}"
        )
        assert value1.dtype == value2.dtype, (
            f"Tensor dtypes differ: {value1.dtype} vs {value2.dtype}"
        )
        assert np.allclose(value1, value2, rtol=float_rtolerance), (
            f"Tensors not close: {value1} vs {value2}"
        )
    # Recurse into nested pydantic models.
    elif isinstance(value1, BaseModel) and isinstance(value2, BaseModel):
        _assert_models_equal(value1, value2, float_rtolerance)
    else:
        assert value1 == value2, f"Values do not match: {value1} vs {value2}"


def _assert_attribute_types_equal(model1, model2):
    """
    Checks that the common attributes in both models have the same type.
    Iterates over the model fields as defined by .__fields__.
    """

    from pydantic import BaseModel

    for field in model1.model_fields_set:
        # If the field exists in both models.
        if hasattr(model2, field):
            type1 = type(getattr(model1, field))
            type2 = type(getattr(model2, field))
            if not issubclass(type1, BaseModel):
                assert type1 == type2, (
                    f"Type mismatch for field '{field}': {type1} vs {type2}"
                )


def _assert_models_values_equal(model1, model2, float_rtolerance=1e-05):
    """
    Checks that two models have the same values.
    It uses compare_values to handle floats (with tolerance), lists, and nested models.
    """
    # Compare using the dict representations. This assumes that both models return the same keys.
    for key in model1.model_fields_set:
        _assert_values_equal(
            getattr(model1, key), getattr(model2, key), float_rtolerance
        )


def _assert_models_equal(model1, model2, float_rtolerance=1e-05):
    """
    Checks that two pydantic models have:
        1. Attributes of the same type.
        2. Close enough values.
    """
    _assert_attribute_types_equal(model1, model2)
    _assert_models_values_equal(model1, model2, float_rtolerance)


def _validate_batched_values(batched_instances, instances):
    import torch

    for attr_name, batched_value in batched_instances.__dict__.items():
        if (
            batched_instances._batch_skip_fields
            and attr_name in batched_instances._batch_skip_fields
        ):
            # Skip fields that are meant to be merged in batches.
            continue
        if (
            batched_instances._batch_merge_fields
            and attr_name in batched_instances._batch_merge_fields
        ):
            # Skip fields that are meant to be merged in batches.
            continue

        if isinstance(batched_value, torch.Tensor):
            continue
        if isinstance(batched_value, list) and isinstance(
            batched_value[0], torch.Tensor
        ):
            continue
        if isinstance(batched_value, BaseDataModel):
            child_instances = [  # the test checks at most 1 level deep lists of models
                [getattr(k, attr_name) for k in instance]
                if isinstance(instance, list)
                else getattr(instance, attr_name)
                for instance in instances
            ]
            _validate_batched_values(batched_value, child_instances)
        else:
            if batched_value is None:
                continue
            assert isinstance(batched_value, list), (
                f"Field {attr_name} is not a list: {batched_value}"
            )
            assert len(batched_value) == len(instances), (
                f"Field {attr_name} has different lengths: {len(batched_value)} != {len(instances)}"
            )
            for i in range(len(batched_value)):
                if isinstance(batched_value[i], list):
                    for j in range(len(batched_value[i])):
                        if isinstance(instances[i], list):
                            _assert_values_equal(
                                batched_value[i][j], getattr(instances[i][j], attr_name)
                            )
                        else:
                            _assert_values_equal(
                                batched_value[i][j], getattr(instances[i], attr_name)[j]
                            )
                else:
                    _assert_values_equal(
                        batched_value[i], getattr(instances[i], attr_name)
                    )
