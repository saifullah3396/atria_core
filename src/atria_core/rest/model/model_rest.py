from functools import partial
from typing import Optional
import uuid

from atria_core.rest.base import RESTBase
from atria_core.schemas.model import (
    Model,
    ModelCreate,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelUpdate,
    ModelVersion,
    ModelVersionCreate,
    ModelVersionUpdate,
)


class RESTModel(RESTBase[Model, ModelCreate, ModelUpdate]):
    def upload(
        self,
        version_tag: str,
        checkpoint_buffer: bytes,
        description: Optional[str] = None,
        is_public: bool = False,
    ) -> None:
        response = self.client.post(
            self._url("upload"),
            data={
                "version_tag": version_tag,
                "is_public": is_public,
                "description": description or "",
            },
            files={
                "model_file": (
                    "model.bin",
                    checkpoint_buffer,
                    "application/octet-stream",
                ),
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to upload model: {response.status_code} - {response.text}"
            )

    def download(self, name: str, version_tag: str, user_id: uuid.UUID) -> bytes:
        response = self.client.post(
            self._url("request_download"),
            json=ModelDownloadRequest(
                name=name, version_tag=version_tag, user_id=user_id
            ).model_dump(),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download model: {response.status_code} - {response.text}"
            )
        download_url = ModelDownloadResponse.model_validate(
            response.json()
        ).download_url
        response = self.client.get(download_url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download model: {response.status_code} - {response.text}"
            )
        return response.content


class RESTModelVersion(RESTBase[ModelVersion, ModelVersionCreate, ModelVersionUpdate]):
    pass


model = partial(RESTModel, model=Model)
model_version = partial(RESTModelVersion, model=ModelVersion)
