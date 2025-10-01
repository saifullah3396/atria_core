import factory
import pyarrow as pa
import pytest

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel
from tests.utilities.common import _assert_values_equal, _validate_batched_values

logger = get_logger(__name__)


class DataModelTestBase:
    throws_error_on_operations = False
    factory: type[factory.Factory]

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        raise NotImplementedError(
            "Child classes must implement the `expected_table_schema` method."
        )

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        raise NotImplementedError(
            "Child classes must implement the `expected_table_schema_flattened` method."
        )

    @pytest.fixture(params=list(range(1)))
    def model_instance(self) -> BaseDataModel:
        """
        Fixture to provide an instance of the BaseDataModel for testing.
        """
        return self.factory.build()

    def test_initialize(self, model_instance: BaseDataModel) -> None:
        """
        Test the initialization of the model instance.
        """
        assert isinstance(model_instance, BaseDataModel), (
            "Model instance is not a BaseDataModel"
        )

    def test_load_unload(self, model_instance: BaseDataModel) -> None:
        """
        Test the load method of the model instance.
        """
        assert not model_instance._is_loaded, "Model instance should be unloaded"
        model_instance.load()
        assert model_instance._is_loaded, "Model instance should be loaded"
        model_instance.unload()
        assert not model_instance._is_loaded, "Model instance should be unloaded again"

    def test_to_from_row(self, model_instance: BaseDataModel) -> None:
        """
        Test the conversion to and from a row representation.
        """
        row = model_instance.to_row()
        assert isinstance(row, dict), "Row representation should be a dictionary"
        new_instance = model_instance.from_row(row)
        assert isinstance(new_instance, BaseDataModel), (
            "New instance should be a BaseDataModel"
        )
        assert new_instance.model_dump() == model_instance.model_dump(), (
            f"New instance does not match original, "
            f"Expected: {model_instance.model_dump()}, Got: {new_instance.model_dump()}"
        )

    def test_schema(self, model_instance: BaseDataModel) -> None:
        """
        Test the schema generation of the model instance.
        """
        schema = model_instance.table_schema()
        expected = self.expected_table_schema()

        def compare_dicts_recursively(dict1, dict2, path=""):
            differences = []
            all_keys = set(dict1.keys()) | set(dict2.keys())

            for key in all_keys:
                current_path = f"{path}.{key}" if path else str(key)

                if key not in dict1:
                    differences.append(f"Missing key in actual: {current_path}")
                elif key not in dict2:
                    differences.append(f"Unexpected key in actual: {current_path}")
                elif dict1[key] != dict2[key]:
                    differences.append(
                        f"Value mismatch at {current_path}: expected {dict2[key]}, got {dict1[key]}"
                    )

            return differences

        differences = compare_dicts_recursively(schema, expected)
        assert schema == expected, (
            "Schema does not match expected schema.\n"
            "Differences found:\n" + "\n".join(differences)
        )

    def test_schema_flattened(self, model_instance: BaseDataModel) -> None:
        """
        Test the schema generation of the model instance.
        """
        schema = model_instance.table_schema_flattened()
        expected = self.expected_table_schema_flattened()
        mismatched_keys = []
        for key in set(schema.keys()) | set(expected.keys()):
            if key not in schema:
                mismatched_keys.append(f"Missing key: {key}")
            elif key not in expected:
                mismatched_keys.append(f"Unexpected key: {key}")
            elif schema[key] != expected[key]:
                mismatched_keys.append(
                    f"Type mismatch for {key}: expected {expected[key]}, got {schema[key]}"
                )

        assert schema == expected, (
            f"Schema does not match expected schema. Mismatched keys: {mismatched_keys}. "
            f"Expected: {expected}, Got: {schema}"
        )

    def test_to_from_tensor(self, model_instance: BaseDataModel) -> None:
        """
        Test the conversion of the model instance to a tensor.
        """

        model_instance.load()

        if self.throws_error_on_operations:
            with pytest.raises(RuntimeError):
                model_instance.load().to_tensor()
            return

        tensor_model = model_instance.to_tensor()
        assert tensor_model is not None, "Tensor conversion returned None"
        roundtrip_model = tensor_model.to_raw()
        assert isinstance(roundtrip_model, model_instance.__class__), (
            "Raw conversion did not return a BaseDataModel"
        )

        _assert_values_equal(roundtrip_model, model_instance)

    def test_to_device(self, model_instance):
        """
        Test the to_device method of the tensor data model.
        """
        import torch

        if self.throws_error_on_operations:
            with pytest.raises(RuntimeError):
                model_instance.load().to_tensor()
            return

        def validate_device(device: str | torch.device):
            instance = model_instance.load().to_tensor().to_device(device)
            for key, value in instance.__dict__.items():
                if isinstance(value, torch.Tensor):
                    assert value.device.type == torch.device(device).type, (
                        f"Field {key} is not on the correct device: {value.device.type} != {torch.device(device).type}"
                    )

        validate_device(torch.device("cpu"))
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available, skipping CUDA tests.")
        validate_device("cuda")
        validate_device(torch.device(0))
        validate_device(torch.device("cuda:0"))
        validate_device(0)

    def test_batched_instances(self, model_instance):
        """
        Test the collation of multiple instances of the child class.
        """
        import torch

        if self.throws_error_on_operations:
            with pytest.raises(RuntimeError):
                model_instance.load().to_tensor()
            return

        instances = [
            model_instance.load().to_tensor(),
            model_instance.load().to_tensor(),
        ]
        model_instance = instances[0].batched(instances)
        assert model_instance._is_batched, (
            "Batched instances should be marked as batched"
        )

        for key, value in model_instance.__dict__.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == len(instances)
                for i in range(1, len(instances)):
                    _assert_values_equal(value[i], getattr(instances[i], key))
        _validate_batched_values(model_instance, instances)
