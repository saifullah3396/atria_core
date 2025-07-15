import pyarrow as pa
import pytest
import torch
from pydantic import ValidationError

from atria_core.types.factory import LabelFactory
from atria_core.types.generic.label import Label
from tests.types.data_model_test_base import DataModelTestBase


class TestLabel(DataModelTestBase):
    """
    Test class for Label.
    """

    factory = LabelFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"name": pa.string(), "value": pa.int64()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"name": pa.string(), "value": pa.int64()}


#########################################################
# Basic Label Tests
#########################################################
@pytest.fixture
def valid_label():
    return Label(value=1, name="Label1")


def test_raw_label_initialization(valid_label):
    assert valid_label.value == 1
    assert valid_label.name == "Label1"


def test_raw_label_value_update(valid_label):
    valid_label.value = 2
    assert valid_label.value == 2


def test_raw_label_name_update(valid_label):
    valid_label.name = "UpdatedLabel"
    assert valid_label.name == "UpdatedLabel"


def test_to_tensor(valid_label: Label) -> None:
    valid_label.load()
    tensor = valid_label.to_tensor()
    assert tensor is not None


###
# Tensor Label Tests
###
@pytest.fixture
def scalar_tensor_label() -> Label:
    """Fixture providing a valid Label with scalar tensor value."""
    return Label(value=1, name="SingleLabel").to_tensor()


def test_tensor_label_creates_valid_scalar_instance(scalar_tensor_label: Label) -> None:
    """Test that Label properly initializes with scalar tensor and name."""
    assert scalar_tensor_label.value.ndim == 0  # Scalar tensor has 0 dimensions
    assert scalar_tensor_label.name == "SingleLabel"


def test_tensor_label_allows_scalar_value_updates(scalar_tensor_label: Label) -> None:
    """Test that tensor value can be updated with another scalar tensor."""
    new_value = torch.tensor(4)
    scalar_tensor_label.value = new_value
    assert torch.equal(scalar_tensor_label.value, new_value)


def test_tensor_label_allows_name_updates(scalar_tensor_label: Label) -> None:
    """Test that label name can be updated to a new string value."""
    new_name = "UpdatedLabel"
    scalar_tensor_label.name = new_name
    assert scalar_tensor_label.name == new_name


def test_tensor_label_rejects_float_dtype() -> None:
    """Test that Label validation fails for non-integer tensor dtypes."""
    with pytest.raises(ValidationError):
        # Float tensors should be rejected
        Label(value=torch.tensor([1.0, 2.5, 3.7]), name="FloatLabel")


def test_tensor_label_rejects_multidimensional_tensors() -> None:
    """Test that Label validation fails for tensors with more than 0 dimensions."""
    with pytest.raises(ValidationError):
        # 2D tensors should be rejected - only scalar tensors allowed
        multi_dim_tensor = torch.tensor([[1, 2], [3, 4]])
        Label(value=multi_dim_tensor, name="MultiDimLabel")


def test_tensor_label_rejects_empty_tensors() -> None:
    """Test that Label validation fails for empty tensor arrays."""
    with pytest.raises(ValidationError):
        # Empty tensors should be rejected
        empty_tensor = torch.tensor([])
        Label(value=empty_tensor, name="EmptyLabel")


def test_tensor_label_equality_comparison() -> None:
    """Test equality comparison between Label instances based on value and name."""
    label1 = Label(value=1, name="Label1").to_tensor()
    label2 = Label(value=1, name="Label1").to_tensor()  # Same as label1
    label3 = Label(value=3, name="Label3").to_tensor()  # Different value

    # Labels with same value and name should have equal components
    assert torch.equal(label1.value, label2.value)
    assert label1.name == label2.name

    # Labels with different values should not be equal
    assert not torch.equal(label1.value, label3.value)


def test_tensor_label_batching_functionality() -> None:
    """Test that Label can be batched into a 1D tensor with repeated values."""
    label = Label(value=10, name="Label1").to_tensor()
    batch_size = 10

    # Create batched version with 10 copies of the same label
    batched_label = label.batched([label] * batch_size)

    # Batched result should be 1D tensor with repeated values
    assert batched_label.value.ndim == 1
    assert torch.equal(batched_label.value, torch.tensor([10] * batch_size))
