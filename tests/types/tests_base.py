import copy
import numbers
from abc import ABC
from typing import List, Type, Union

import pytest
import torch
from pydantic import Field

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel, BatchedBaseDataModel
from tests.utilities import _compare_values

logger = get_logger(__name__)


class BaseDataModelTestBase(ABC):
    """
    A base test class for testing all child classes of BaseDataModel.

    Child test classes must implement the `get_model_instance` method to provide
    an instance of the child class being tested.
    """

    def batched_model(self) -> Type[BatchedBaseDataModel]:
        """
        Return the batched model class to be tested.
        """
        raise NotImplementedError(
            "Child classes must implement the `batched_model` method."
        )

    def tensor_fields(self) -> List[str]:
        """
        Return an instance of the child class to be tested.
        """
        raise NotImplementedError(
            "Child classes must implement the `get_model_instance` method."
        )

    def test_to_tensor(self, model_instance):
        """
        Test the load method of the child class.
        """
        instance = model_instance.to_tensor()
        assert instance._is_tensor is True
        for name in self.tensor_fields():
            assert isinstance(
                getattr(instance, name), torch.Tensor
            ), f"Field {name} is not a tensor: {getattr(instance, name)}"

    def test_serialize(self, model_instance):
        """
        Test the serialize method of the child class.
        """
        instance = model_instance.to_tensor()
        seralized_instance = instance.model_dump()
        for name in self.tensor_fields():
            assert isinstance(
                seralized_instance[name], (numbers.Number, list)
            ), f"Field {name} is not a list: {seralized_instance[name]}"
        assert isinstance(
            seralized_instance, dict
        ), "Serialized instance is not a dictionary"

    def test_to_device(self, model_instance):
        """
        Test the to_device method of the child class.
        """
        with pytest.raises(AssertionError):
            model_instance.to_device(0)

        def validate_device(device: Union[str, torch.device]):
            instance = model_instance.to_tensor().to_device(device)
            for name in self.tensor_fields():
                value = getattr(instance, name)
                assert isinstance(
                    value, torch.Tensor
                ), f"Field {name} is not a tensor: {value}"
                assert (
                    value.device.type == torch.device(device).type
                ), f"Field {name} is not on the correct device: {value.device.type} != {torch.device(device).type}"

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

        batched_instances = [model_instance, model_instance]
        with pytest.raises(AssertionError):
            batched_instances[0].batched(batched_instances)
        instances = [x.to_tensor() for x in copy.deepcopy(batched_instances)]
        batched_instances = instances[0].batched(instances)
        assert isinstance(batched_instances, BaseDataModel)
        assert isinstance(
            batched_instances,
            self.batched_model(),
        )

        def _validate_batched_values(batched_instances, instances):
            for attr_name, batched_value in batched_instances.__dict__.items():
                if attr_name == "batch_size":
                    continue
                if isinstance(batched_value, torch.Tensor):
                    continue
                if isinstance(batched_value, list) and isinstance(
                    batched_value[0], torch.Tensor
                ):
                    continue
                if isinstance(batched_value, BaseDataModel):
                    child_instances = [
                        getattr(instance, attr_name) for instance in instances
                    ]
                    _validate_batched_values(batched_value, child_instances)
                else:
                    if batched_value is None:
                        continue
                    assert isinstance(
                        batched_value, list
                    ), f"Field {attr_name} is not a list: {batched_value}"
                    assert len(batched_value) == len(
                        instances
                    ), f"Field {attr_name} has different lengths: {len(batched_value)} != {len(instances)}"
                    for i in range(len(batched_value)):
                        if isinstance(batched_value[i], list):
                            for j in range(len(batched_value[i])):
                                if isinstance(instances[i], list):
                                    _compare_values(
                                        batched_value[i][j],
                                        getattr(instances[i][j], attr_name),
                                    )
                                else:
                                    _compare_values(
                                        batched_value[i][j],
                                        getattr(instances[i], attr_name)[j],
                                    )
                        else:
                            _compare_values(
                                batched_value[i], getattr(instances[i], attr_name)
                            )

        for name in self.tensor_fields():
            value = getattr(batched_instances, name)
            assert isinstance(value, torch.Tensor)
            assert value.shape[0] == len(instances)
            for i in range(1, len(instances)):
                _compare_values(value[i], getattr(instances[i], name))
        _validate_batched_values(batched_instances, instances)

        with pytest.raises(RuntimeError):
            batched_instances.model_dump()
            batched_instances.model_dump_json()


class MockDataModel(BaseDataModel):
    required_integer_attribute: int
    required_integer_list_attribute: List[int]
    integer_attribute: int = 0
    float_attribute: float = 0.0
    string_attribute: str = ""
    list_attribute: List[int] = Field(default_factory=list[int])
    integer_list_attribute: List[int] = Field(default_factory=list[int])
    float_list_attribute: List[float] = Field(default_factory=list[float])
    string_list_attribute: List[str] = Field(default_factory=list[str])
    attribute1: torch.Tensor = 0
    attribute2: torch.Tensor = 0.0
    tensor_attribute: torch.Tensor = torch.tensor(0.0)
    tensor_list_attribute: torch.Tensor = Field(
        default_factory=lambda: [torch.tensor(0.0)]
    )
    list_of_tensors_attribute: torch.Tensor = Field(
        default_factory=lambda: [
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ]
    )
    variable_tensor_list_attribute: torch.Tensor = Field(
        default_factory=lambda: [
            torch.tensor(0.0),
        ]
    )

    def _load(self):
        """
        Custom implementation of the load method.
        """
        self.attribute1 = 42
        self.attribute2 = 3.14
        self.tensor_attribute = torch.tensor([1.0, 2.0, 3.0])
        self.tensor_list_attribute = [
            torch.tensor([4.0, 5.0]),
            torch.tensor([6.0, 7.0]),
        ]
        self.variable_tensor_list_attribute = [
            torch.tensor([4.0, 5.0]),
            torch.tensor([6.0, 7.0, 8.0]),
        ]


class BatchedMockDataModel(BatchedBaseDataModel):
    required_integer_attribute: List[int]
    required_integer_list_attribute: List[List[int]]
    integer_attribute: List[int]
    float_attribute: List[float]
    string_attribute: List[str]
    list_attribute: List[List[int]]
    integer_list_attribute: List[List[int]]
    float_list_attribute: List[List[float]]
    string_list_attribute: List[List[str]]
    attribute1: torch.Tensor
    attribute2: torch.Tensor
    tensor_attribute: torch.Tensor
    tensor_list_attribute: torch.Tensor
    list_of_tensors_attribute: torch.Tensor
    variable_tensor_list_attribute: torch.Tensor


class MockDataModelChild(MockDataModel):
    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedMockDataModelChild":
        return BatchedMockDataModelChild(**kwargs)


class BatchedMockDataModelChild(BatchedMockDataModel):
    pass


class MockDataModelParent(MockDataModel):
    example_data_model_child: MockDataModelChild

    @classmethod
    def batched_construct(cls, **kwargs) -> "BatchedMockDataModelParent":
        return BatchedMockDataModelParent(**kwargs)


class BatchedMockDataModelParent(BatchedMockDataModel):
    example_data_model_child: BatchedMockDataModelChild
    example_data_model_child: BatchedMockDataModelChild
    example_data_model_child: BatchedMockDataModelChild
    example_data_model_child: BatchedMockDataModelChild
