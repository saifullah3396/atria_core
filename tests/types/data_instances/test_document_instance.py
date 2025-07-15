import pyarrow as pa

from atria_core.types.data_instance.document_instance import DocumentInstance
from atria_core.types.factory import DocumentInstanceFactory
from tests.types.data_model_test_base import DataModelTestBase
from tests.utilities.common import _assert_values_equal


class TestDocumentInstance(DataModelTestBase):
    """
    Test class
    """

    factory = DocumentInstanceFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "index": pa.int64(),
            "sample_id": pa.string(),
            "page_id": pa.int64(),
            "total_num_pages": pa.int64(),
            "image": {
                "file_path": pa.string(),
                "content": pa.binary(),
                "source_width": pa.int64(),
                "source_height": pa.int64(),
            },
            "gt": {
                "classification": pa.string(),
                "ser": pa.string(),
                "ocr": pa.string(),
                "qa": pa.string(),
                "vqa": pa.string(),
                "layout": pa.string(),
            },
            "ocr": {
                "file_path": pa.string(),
                "content": pa.binary(),
                "type": pa.string(),
            },
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "index": pa.int64(),
            "sample_id": pa.string(),
            "page_id": pa.int64(),
            "total_num_pages": pa.int64(),
            "image_file_path": pa.string(),
            "image_content": pa.binary(),
            "image_source_width": pa.int64(),
            "image_source_height": pa.int64(),
            "gt_classification": pa.string(),
            "gt_ser": pa.string(),
            "gt_ocr": pa.string(),
            "gt_qa": pa.string(),
            "gt_vqa": pa.string(),
            "gt_layout": pa.string(),
            "ocr_file_path": pa.string(),
            "ocr_content": pa.binary(),
            "ocr_type": pa.string(),
        }

    def test_to_from_tensor(self, model_instance: DocumentInstance) -> None:
        """
        Test the conversion of the model instance to a tensor.
        """
        model_instance.load()
        tensor_model = model_instance.to_tensor()
        assert tensor_model is not None, "Tensor conversion returned None"
        roundtrip_model = tensor_model.to_raw()
        assert isinstance(roundtrip_model, model_instance.__class__), (
            "Raw conversion did not return a BaseDataModel"
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
