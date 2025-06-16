from enum import Enum
from typing import TYPE_CHECKING, Dict

from fastapi import HTTPException
from pydantic import computed_field

from atria_core.schemas.base import BaseDataInstanceStorageSchema
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.datasets.splits import DatasetSplitType

if TYPE_CHECKING:
    from atriax.storage.dataset.document_instance_storage import DocumentInstanceStorage


class OCRStatus(str, Enum):
    available = "available"
    unavailable = "unavailable"


class DocumentInstanceBase(BaseDataInstance):
    branch_name: str
    split: DatasetSplitType
    sample_id: str
    ocr_status: OCRStatus = OCRStatus.unavailable


class DocumentInstance(DocumentInstanceBase, BaseDataInstanceStorageSchema):
    def get_storage_instance(self) -> "DocumentInstanceStorage":
        from atriax import storage

        return storage.document_instance

    @computed_field
    @property
    def ocr_urls(self) -> Dict[str, str]:
        pass

        if self.storage_objects:
            ocr_urls = {}
            for obj in self.storage_objects:
                if obj.object_key.startswith(
                    self.get_storage_instance().__default_ocr_path__
                ):
                    ocr_urls[
                        obj.object_key.replace(
                            self.get_storage_instance().__default_ocr_path__, ""
                        )
                    ] = obj.presigned_url
            return ocr_urls if ocr_urls else []
        return []

    async def fetch_ocr(self) -> Dict[str, str]:
        ocr = {}
        for key, url in self.ocr_urls.items():
            try:
                ocr[key] = await self.fetch_object(url)
                ocr[key] = ocr[key].decode("utf-8")
            except HTTPException as e:
                if e.status_code == 404:
                    ocr[key] = None
                    continue
                raise e
        return ocr
