# import pytest

# from atria_core.types.base.tensor_data_model import BatchedBaseDataModel
# from atria_core.types.data_instance.tokenized_object import (
#     BatchedTokenizedObjectInstance,
#     TokenizedObjectInstance,
# )
# from atria_core.types.generic._image import Image
# from atria_core.types.generic._label import Label
# from tests.types.factory import TokenizedObjectInstanceFactory
# from tests.types.tests_base import BaseDataModelTestBase


# class TestTokenizedObjectInstance(BaseDataModelTestBase):
#     @pytest.fixture
#     def model_instance(self) -> TokenizedObjectInstance:
#         return TokenizedObjectInstanceFactory.build()

#     def batched_model(self) -> type[BatchedBaseDataModel]:
#         return BatchedTokenizedObjectInstance

#     def tensor_fields(self) -> list[str]:
#         return []

#     def test_initialization(self, model_instance):
#         tokenized_object_instance = model_instance.to_tensor()

#         assert isinstance(
#             tokenized_object_instance.image, Image
#         ), "Image attribute is not an instance of Image"

#         # Check if the label attribute is an instance of Label
#         assert isinstance(
#             tokenized_object_instance.label, Label
#         ), "Label attribute is not an instance of Label"

#         assert tokenized_object_instance.image.content.shape == (
#             3,
#             256,
#             256,
#         ), "Image content shape mismatch"
#         assert tokenized_object_instance.image.shape == (
#             3,
#             256,
#             256,
#         ), "Image size mismatch"

#         assert tokenized_object_instance.token_ids.shape == (
#             16,
#         ), "Token IDs shape mismatch"
#         assert tokenized_object_instance.word_ids.shape == (
#             16,
#         ), "Word IDs shape mismatch"
#         assert tokenized_object_instance.token_labels.shape == (
#             16,
#         ), "Token labels shape mismatch"
#         assert tokenized_object_instance.token_type_ids.shape == (
#             16,
#         ), "Token type IDs shape mismatch"
#         assert tokenized_object_instance.attention_mask.shape == (
#             16,
#         ), "Attention mask shape mismatch"
#         assert tokenized_object_instance.token_bboxes.shape == (
#             16,
#             4,
#         ), "Token bounding boxes shape mismatch"
#         assert tokenized_object_instance.sequence_ids.shape == (
#             16,
#         ), "Sequence IDs shape mismatch"
