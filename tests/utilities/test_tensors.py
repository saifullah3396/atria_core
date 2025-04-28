import numpy as np
import torch

from atria_core.utilities.tensors import _convert_to_tensor


def test_convert_list_of_numbers():
    """Test conversion of a list of numbers to a tensor."""
    result = _convert_to_tensor([1, 2, 3])
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1, 2, 3]))


def test_convert_single_number():
    """Test conversion of a single number to a tensor."""
    result = _convert_to_tensor(5)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor(5))


def test_convert_list_of_tensors():
    """Test conversion of a list of tensors to a stacked tensor."""
    tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    result = _convert_to_tensor(tensors)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.stack(tensors))


def test_convert_list_of_numpy_arrays():
    """Test conversion of a list of NumPy arrays to a tensor."""
    arrays = [np.array([1, 2]), np.array([3, 4])]
    result = _convert_to_tensor(arrays)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([[1, 2], [3, 4]]))


def test_convert_single_numpy_array():
    """Test conversion of a single NumPy array to a tensor."""
    array = np.array([1, 2, 3])
    result = _convert_to_tensor(array)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1, 2, 3]))


def test_convert_list_of_strings():
    """Test that a list of strings remains unchanged."""
    result = _convert_to_tensor(["a", "b", "c"])
    assert isinstance(result, list)
    assert result == ["a", "b", "c"]


def test_convert_nested_list_of_numbers():
    """Test conversion of a nested list of numbers to tensors."""
    result = _convert_to_tensor([[1, 2], [3, 4]])
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([[1, 2], [3, 4]]))


def test_convert_nested_list_of_numbers_var():
    """Test conversion of a nested list of numbers to tensors."""
    result = _convert_to_tensor([[1, 2], [3, 4, 5]])
    assert isinstance(result, list)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert torch.equal(result[0], torch.tensor([1, 2]))
    assert torch.equal(result[1], torch.tensor([3, 4, 5]))


def test_convert_nested_list_of_tensors():
    """Test conversion of a nested list of tensors to tensors."""
    result = _convert_to_tensor([[torch.tensor([1, 2])], [torch.tensor([3, 4])]])
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([[[1, 2]], [[3, 4]]]))


def test_convert_nested_list_of_varible_size_tensors1():
    """Test conversion of a nested list of tensors to tensors."""
    result = _convert_to_tensor([torch.tensor([1, 2]), torch.tensor([3, 4, 5])])
    assert isinstance(result, list)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert torch.equal(result[0], torch.tensor([1, 2]))
    assert torch.equal(result[1], torch.tensor([3, 4, 5]))


def test_convert_nested_list_of_varible_size_tensors2():
    """Test conversion of a nested list of tensors to tensors."""
    result = _convert_to_tensor([[torch.tensor([1, 2])], [torch.tensor([3, 4, 5])]])
    assert isinstance(result, list)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert torch.equal(result[0], torch.tensor([[1, 2]]))
    assert torch.equal(result[1], torch.tensor([[3, 4, 5]]))


def test_convert_nested_list_of_varible_size_tensors3():
    """Test conversion of a nested list of tensors to tensors."""
    result = _convert_to_tensor(
        [
            [torch.tensor([1, 2]), torch.tensor([1, 2, 3])],
            [torch.tensor([4, 5]), torch.tensor([6, 7, 8])],
        ]
    )
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)
    assert torch.equal(result[0][0], torch.tensor([1, 2]))
    assert torch.equal(result[0][1], torch.tensor([1, 2, 3]))
    assert torch.equal(result[1][0], torch.tensor([4, 5]))
    assert torch.equal(result[1][1], torch.tensor([6, 7, 8]))


def test_convert_list_with_empty_nested_list():
    """Test handling of a list containing an empty nested list."""
    assert _convert_to_tensor([]).shape == torch.Size([0])
    assert _convert_to_tensor([[]]).shape == torch.Size([1, 0])
    converted = _convert_to_tensor([[], [1, 2, 3]])
    assert converted[0].shape == torch.Size([0])
    assert converted[1].shape == torch.Size([3])
