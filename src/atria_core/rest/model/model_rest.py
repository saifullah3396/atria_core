from functools import partial
import io
import json
from typing import Optional
import uuid

import httpx
import torch

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
    pass


class RESTModelVersion(RESTBase[ModelVersion, ModelVersionCreate, ModelVersionUpdate]):
    def upload(
        self,
        version_tag: str,
        checkpoint: bytes,
        description: Optional[str] = None,
        is_public: bool = False,
    ) -> None:
        checkpoint_buffer = io.BytesIO()
        torch.save(checkpoint, checkpoint_buffer)
        checkpoint_buffer.seek(0)
        response = self.client.post(
            self._url("upload"),
            data={
                "version_tag": version_tag,
                "is_public": is_public,
                "description": description or "",
            },
            files={
                "model_checkpoint": (
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

    def download(self, download_request: ModelDownloadRequest) -> bytes:
        response = self.client.post(
            self._url("request_download"),
            json=download_request.model_dump(),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download model: {response.status_code} - {response.text}"
            )
        download_url = ModelDownloadResponse.model_validate(
            response.json()
        ).download_url
        with httpx.Client() as client:
            response = client.get(download_url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download model: {response.status_code} - {response.text}"
            )
        return response.content


model = partial(RESTModel, model=Model)
model_version = partial(
    RESTModelVersion, model=ModelVersion, resource_path="model_version"
)
