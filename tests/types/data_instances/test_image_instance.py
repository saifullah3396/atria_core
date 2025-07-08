import pyarrow as pa

from atria_core.types.data_instance._raw.image_instance import ImageInstance
from atria_core.types.factory import ImageInstanceFactory
from tests.types.data_model_test_base import DataModelTestBase
from tests.utilities.common import _assert_values_equal


class TestImageInstance(DataModelTestBase):
    """
    Test class for Label.
    """

    factory = ImageInstanceFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "index": pa.int64(),
            "sample_id": pa.string(),
            "image": {
                "file_path": pa.string(),
                "content": pa.binary(),
                "width": pa.int64(),
                "height": pa.int64(),
            },
            "gt": {
                "classification": pa.string(),
                "ser": pa.string(),
                "ocr": pa.string(),
                "qa": pa.string(),
                "vqa": pa.string(),
                "layout": pa.string(),
            },
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the RawDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "index": pa.int64(),
            "sample_id": pa.string(),
            "image_file_path": pa.string(),
            "image_content": pa.binary(),
            "image_width": pa.int64(),
            "image_height": pa.int64(),
            "gt_classification": pa.string(),
            "gt_ser": pa.string(),
            "gt_ocr": pa.string(),
            "gt_qa": pa.string(),
            "gt_vqa": pa.string(),
            "gt_layout": pa.string(),
        }

    def test_to_from_tensor(self, model_instance: ImageInstance) -> None:
        """
        Test the conversion of the model instance to a tensor.
        """
        model_instance.load()
        tensor_model = model_instance.to_tensor()
        assert tensor_model is not None, "Tensor conversion returned None"
        assert isinstance(tensor_model, model_instance.tensor_data_model()), (
            "Tensor conversion did not return the expected tensor model type"
        )
        roundtrip_model = tensor_model.to_raw()
        assert isinstance(roundtrip_model, model_instance.__class__), (
            "Raw conversion did not return a RawDataModel"
        )

        set_fields = roundtrip_model.image.model_fields_set
        roundtrip_model_image = roundtrip_model.image.model_dump(include=set_fields)
        original_data_image = model_instance.image.model_dump(include=set_fields)
        _assert_values_equal(roundtrip_model_image, original_data_image)
        for field in model_instance.model_fields:
            if field == "image":
                continue
            _assert_values_equal(
                getattr(roundtrip_model, field), getattr(model_instance, field)
            )
