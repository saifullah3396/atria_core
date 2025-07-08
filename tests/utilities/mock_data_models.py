import factory
import torch
from pydantic import Field

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import RawDataModel, TensorDataModel
from atria_core.types.typing.common import (
    FloatField,
    IntField,
    ListFloatField,
    ListIntField,
    ListStrField,
    StrField,
)

logger = get_logger(__name__)


class MockRawDataModel(RawDataModel):
    required_integer_attribute: IntField
    required_integer_list_attribute: ListIntField
    integer_attribute: IntField = 0
    float_attribute: FloatField = 0.0
    string_attribute: StrField = ""
    list_attribute: ListIntField = Field(default_factory=ListIntField)
    integer_list_attribute: ListIntField = Field(default_factory=ListIntField)
    float_list_attribute: ListFloatField = Field(default_factory=ListFloatField)
    string_list_attribute: ListStrField = Field(default_factory=ListStrField)

    @classmethod
    def tensor_data_model(cls):
        return MockTensorDataModel


class MockDataModelChild(MockRawDataModel):
    @classmethod
    def tensor_data_model(cls):
        return MockTensorDataModelChild


class MockDataModelParent(MockRawDataModel):
    example_data_model_child: MockDataModelChild

    @classmethod
    def tensor_data_model(cls):
        return MockTensorDataModelParent


class MockTensorDataModel(TensorDataModel):
    required_integer_attribute: torch.Tensor
    required_integer_list_attribute: torch.Tensor
    integer_attribute: torch.Tensor = 0
    float_attribute: torch.Tensor = 0.0
    string_attribute: str = ""
    list_attribute: torch.Tensor
    integer_list_attribute: torch.Tensor
    float_list_attribute: torch.Tensor
    string_list_attribute: list[str]

    @classmethod
    def raw_data_model(cls) -> type[RawDataModel]:
        return MockRawDataModel


class MockTensorDataModelChild(MockTensorDataModel):
    @classmethod
    def raw_data_model(cls) -> type[RawDataModel]:
        return MockDataModelChild


class MockTensorDataModelParent(MockTensorDataModel):
    example_data_model_child: MockTensorDataModelChild

    @classmethod
    def raw_data_model(cls) -> type[RawDataModel]:
        return MockDataModelParent


class MockRawDataModelFactory(factory.Factory):
    class Meta:
        model = MockRawDataModel

    required_integer_attribute = factory.Faker("random_int", min=1, max=100)
    required_integer_list_attribute = factory.List(
        [factory.Faker("random_int", min=1, max=100) for _ in range(3)]
    )
    integer_attribute = factory.Faker("random_int", min=0, max=50)
    float_attribute = factory.Faker("pyfloat", positive=True)
    string_attribute = factory.Faker("word")
    list_attribute = factory.List(
        [factory.Faker("random_int", min=1, max=100) for _ in range(2)]
    )
    integer_list_attribute = factory.List(
        [factory.Faker("random_int", min=1, max=100) for _ in range(2)]
    )
    float_list_attribute = factory.List(
        [factory.Faker("pyfloat", positive=True) for _ in range(2)]
    )
    string_list_attribute = factory.List([factory.Faker("word") for _ in range(2)])


class MockDataModelChildFactory(MockRawDataModelFactory):
    class Meta:
        model = MockDataModelChild


class MockDataModelParentFactory(MockRawDataModelFactory):
    class Meta:
        model = MockDataModelParent

    example_data_model_child = factory.SubFactory(MockDataModelChildFactory)
