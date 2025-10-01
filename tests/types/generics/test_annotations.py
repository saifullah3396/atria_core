import pyarrow as pa

from atria_core.types.factory import (
    ClassificationAnnotationFactory,
    EntityLabelingAnnotationFactory,
    ExtractiveQAAnnotationFactory,
    LayoutAnalysisAnnotationFactory,
)
from tests.types.data_model_test_base import DataModelTestBase


class TestClassificationAnnotation(DataModelTestBase):
    """
    Test class for ClassificationAnnotation.
    """

    factory = ClassificationAnnotationFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"label": {"name": pa.string(), "value": pa.int64()}}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"label_name": pa.string(), "label_value": pa.int64()}


class TestEntityLabelingAnnotation(DataModelTestBase):
    """
    Test class for TestEntityLabeling.
    """

    factory = EntityLabelingAnnotationFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "word_labels": {
                "name": pa.list_(pa.string()),
                "value": pa.list_(pa.int64()),
            }
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "word_labels_name": pa.list_(pa.string()),
            "word_labels_value": pa.list_(pa.int64()),
        }


class TestExtractiveQAAnnotation(DataModelTestBase):
    """
    Test class for TestEntityLabeling.
    """

    factory = ExtractiveQAAnnotationFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"qa_pairs": pa.string()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"qa_pairs": pa.string()}


class TestLayoutAnalysisAnnotation(DataModelTestBase):
    """
    Test class for TestEntityLabeling.
    """

    factory = LayoutAnalysisAnnotationFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"annotated_objects": pa.string()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"annotated_objects": pa.string()}
