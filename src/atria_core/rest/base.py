import uuid
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class RESTBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(
        self,
        *,
        client: httpx.Client,
        model: Type[ModelType],
    ):
        self.client = client
        self.model = model
        self.resource_path = model.__name__.lower()

    def _url(self, suffix: str = "") -> str:
        return f"/rest/v1/{self.resource_path}/{suffix}".rstrip("/")

    def _serialize_filters(self, filters: Dict[str, Any]) -> Dict[str, str]:
        return {
            key: str(value) if isinstance(value, (uuid.UUID, int, float)) else value
            for key, value in filters.items()
        }

    def get(self, *, id: uuid.UUID) -> Optional[ModelType]:
        response = self.client.get(self._url("get"), params={"id": str(id)})
        if response.status_code == 200:
            return self.model.model_validate(response.json())
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def filter(self, **filters: Any) -> Optional[ModelType]:
        response = self.client.get(
            self._url("filter"), params=self._serialize_filters(filters)
        )
        if response.status_code == 200 and response.json():
            return self.model.model_validate(response.json())
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def count(self, **filters: Any) -> int:
        response = self.client.get(
            self._url("count"), params=self._serialize_filters(filters)
        )
        if response.status_code == 200:
            return response.json().get("count", 0)
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def list(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "created_at",
        order: str = "desc",
        **filters: Any,
    ) -> List[ModelType]:
        params = {
            **self._serialize_filters(filters),
            "skip": skip,
            "limit": limit,
            "order_by": order_by,
            "order": order,
        }
        response = self.client.get(self._url("list"), params=params)
        if response.status_code == 200:
            return [self.model.model_validate(item) for item in response.json()]
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def create(self, *, obj_in: CreateSchemaType, **kwargs: Any) -> Optional[ModelType]:
        payload = obj_in.model_dump()
        payload.update(self._serialize_filters(kwargs))
        response = self.client.post(self._url("create"), json=payload)
        if response.status_code in (200, 201):
            return self.model.model_validate(response.json())
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def upsert(self, *, obj_in: CreateSchemaType, **kwargs: Any) -> Optional[ModelType]:
        payload = obj_in.model_dump()
        payload.update(self._serialize_filters(kwargs))
        response = self.client.put(self._url("upsert"), json=payload)
        if response.status_code in (200, 201):
            return self.model.model_validate(response.json())
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def update(self, *, id: uuid.UUID, obj_in: UpdateSchemaType) -> Optional[ModelType]:
        update_data = obj_in.model_dump(exclude_unset=True)
        response = self.client.patch(
            self._url("update"), params={"id": str(id)}, json=update_data
        )
        if response.status_code == 200:
            return self.model.model_validate(response.json())
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def delete(self, *, id: uuid.UUID) -> Optional[ModelType]:
        response = self.client.delete(self._url("delete"), params={"id": str(id)})
        if response.status_code == 200:
            return self.model.model_validate(response.json())
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )
