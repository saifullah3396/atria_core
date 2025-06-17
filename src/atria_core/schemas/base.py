import enum
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from fastapi import HTTPException
from httpx import AsyncClient
import lakefs
from pydantic import BaseModel, ConfigDict, computed_field

from atria_core.schemas.utils import SerializableDateTime, SerializableUUID
from atria_core.types.generic.ground_truth import GroundTruth

if TYPE_CHECKING:
    from atria_core.types.data_instance.base import BaseDataInstance
    from atriax.storage.dataset.document_instance_storage import DocumentInstanceStorage
    from atriax.storage.dataset.image_instance_storage import ImageInstanceStorage


class OptionalModel(BaseModel):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        for field in cls.model_fields.values():
            field.annotation = field.annotation | None  # <- for valid JsonSchema
            field.default = None

        cls.model_rebuild(force=True)


class LakeFSBranchSummary(BaseModel):
    name: str
    last_commit_id: str
    last_committer: str
    last_commit_message: str
    last_commit_date: int

    @classmethod
    def from_lakefs_branch(cls, branch: lakefs.Branch) -> "LakeFSBranchSummary":
        commit = branch.head.get_commit()
        return cls(
            name=branch.id,
            last_commit_id=commit.id,
            # last_committer=commit.committer,
            last_committer=commit.metadata.get("committer", "unknown"),
            last_commit_message=commit.message,
            last_commit_date=commit.creation_date,
        )


class LakeFSMetadataObject(BaseModel):
    main_branch: str
    branches: List[LakeFSBranchSummary]


class StorageObject(BaseModel):
    base_path: str
    object_key: str
    ext: str
    presigned_url: str | None = None
    size: int = 0

    @computed_field
    @property
    def path(self) -> str:
        return f"{self.base_path}/{self.object_key}{self.ext}"


class LakeFSStorageObject(BaseModel):
    physical_address: str | None = None
    object_key: str
    ext: str
    presigned_url: str | None = None
    size: int = 0
    modified: int = 0
    type: str = "file"

    @computed_field
    @property
    def path(self) -> str:
        return f"{self.object_key}{self.ext}"


class LakeFSStoragePaginatedObjects(BaseModel):
    objects: List[LakeFSStorageObject]
    next_after: Optional[str] = None


class DataInstanceType(str, enum.Enum):
    image_instance = "image_instance"
    document_instance = "document_instance"

    @classmethod
    def from_data_model(
        self, data_model: Type["BaseDataInstance"]
    ) -> "DataInstanceType":
        """
        Convert a data model to a DataInstanceType.
        Args:
            data_model (Type[BaseDataInstance]): The data model class.
        Returns:
            DataInstanceType: The corresponding DataInstanceType.
        """
        from atria_core.types.data_instance.document import (
            DocumentInstance as AtriaDocumentInstance,
        )
        from atria_core.types.data_instance.image import (
            ImageInstance as AtriaImageInstance,
        )

        if issubclass(data_model, AtriaDocumentInstance):
            return DataInstanceType.document_instance
        elif issubclass(data_model, AtriaImageInstance):
            return DataInstanceType.image_instance
        else:
            raise ValueError(f"Unsupported data model: {data_model.__name__}")

    @classmethod
    def to_data_model(
        cls, data_instance_type: "DataInstanceType"
    ) -> Type["BaseDataInstance"]:
        """
        Convert a DataInstanceType to a data model.
        Args:
            data_instance_type (DataInstanceType): The DataInstanceType.
        Returns:
            Type[BaseDataInstance]: The corresponding data model class.
        """
        from atria_core.types.data_instance.document import (
            DocumentInstance as AtriaDocumentInstance,
        )
        from atria_core.types.data_instance.image import (
            ImageInstance as AtriaImageInstance,
        )

        if data_instance_type == DataInstanceType.document_instance:
            return AtriaDocumentInstance
        elif data_instance_type == DataInstanceType.image_instance:
            return AtriaImageInstance
        else:
            raise ValueError(f"Unsupported data instance type: {data_instance_type}")


class BaseDatabaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")
    id: SerializableUUID
    created_at: SerializableDateTime
    updated_at: SerializableDateTime


class BaseS3StorageDatabaseSchema(BaseDatabaseSchema):
    storage_objects: Optional[List[Union[StorageObject, LakeFSStorageObject]]] = None
    total_size: int = 0

    async def fetch_object(self, url: str) -> bytes:
        if not url:
            raise HTTPException(
                status_code=404,
                detail=f"Object {url} not found in dataset {self.name}.",
            )
        async with AsyncClient() as client:
            response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to load model from {url}: {response.text}",
            )
        return response.content


class BaseStorageDatabaseSchema(BaseDatabaseSchema):
    storage_objects: Optional[List[Union[StorageObject, LakeFSStorageObject]]] = None
    storage_metadata: LakeFSMetadataObject
    total_size: int = 0

    async def fetch_object(self, url: str) -> bytes:
        if not url:
            raise HTTPException(
                status_code=404,
                detail=f"Object {url} not found in dataset {self.name}.",
            )
        async with AsyncClient() as client:
            response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to load model from {url}: {response.text}",
            )
        return response.content


class BaseDataInstanceStorageSchema(BaseStorageDatabaseSchema):
    dataset_id: SerializableUUID

    def get_storage_instance(
        self,
    ) -> Union["ImageInstanceStorage", "DocumentInstanceStorage"]:
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the storage instance."
        )

    @computed_field
    @property
    def image_url(self) -> str | None:

        if self.storage_objects:
            return next(
                (
                    obj.presigned_url
                    for obj in self.storage_objects
                    if obj.object_key
                    == self.get_storage_instance().__default_image_path__
                ),
                None,
            )
        return None

    @computed_field
    @property
    def gt_urls(self) -> Dict[str, str]:

        if self.storage_objects:
            gt_urls = {}
            for obj in self.storage_objects:
                if obj.object_key.startswith(
                    self.get_storage_instance().__default_gt_path__
                ):
                    gt_urls[
                        obj.object_key.replace(
                            self.get_storage_instance().__default_gt_path__, ""
                        )
                    ] = obj.presigned_url
            return gt_urls if gt_urls else []
        return []

    async def fetch_image(self, return_pil_image: bool) -> str | None:
        try:
            image = await self.fetch_object(self.image_url)
            if return_pil_image:
                import io

                from PIL import Image

                return Image.open(io.BytesIO(image))
            else:
                return image
        except HTTPException as e:
            if e.status_code == 404:
                return None
            raise e

    async def fetch_gt(self) -> GroundTruth:
        gt = {}
        for key, url in self.gt_urls.items():
            try:
                gt[key] = json.loads(await self.fetch_object(url))
            except HTTPException as e:
                if e.status_code == 404:
                    gt[key] = None
                    continue
                raise e
        return GroundTruth.model_validate(gt)
