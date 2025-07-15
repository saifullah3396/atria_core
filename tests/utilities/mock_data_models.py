import factory
from pydantic import Field

from atria_core.logger.logger import get_logger
from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import (
    FloatField,
    IntField,
    ListFloatField,
    ListIntField,
    ListStrField,
    StrField,
)

logger = get_logger(__name__)


class MockBaseDataModel(BaseDataModel):
    required_integer_attribute: IntField
    required_integer_list_attribute: ListIntField
    integer_attribute: IntField = 0
    float_attribute: FloatField = 0.0
    string_attribute: StrField = ""
    list_attribute: ListIntField = Field(default_factory=ListIntField)
    integer_list_attribute: ListIntField = Field(default_factory=ListIntField)
    float_list_attribute: ListFloatField = Field(default_factory=ListFloatField)
    string_list_attribute: ListStrField = Field(default_factory=ListStrField)


class MockDataModelChild(MockBaseDataModel):
    pass


class MockDataModelParent(MockBaseDataModel):
    example_data_model_child: MockDataModelChild


class MockBaseDataModelFactory(factory.Factory):
    class Meta:
        model = MockBaseDataModel

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


class MockDataModelChildFactory(MockBaseDataModelFactory):
    class Meta:
        model = MockDataModelChild


class MockDataModelParentFactory(MockBaseDataModelFactory):
    class Meta:
        model = MockDataModelParent

    example_data_model_child = factory.SubFactory(MockDataModelChildFactory)
