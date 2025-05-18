from functools import partial
import io
import json
import os
from typing import Optional
import uuid

import httpx
import requests
import torch
import tqdm

from atria.data.datasets.downloads.file_downloader import HTTPDownloader
from atria_core.logger.logger import get_logger
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

logger = get_logger(__name__)


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

    def _download_url(self, url: str, destination_path: str = None) -> bytes | None:
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=10,
            )
            if response.status_code == 200:
                total_size = int(response.headers.get("Content-Length", 0))
                block_size = 8192  # 8 KB
                if destination_path:
                    with (
                        open(destination_path, "wb") as f,
                        tqdm.tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=os.path.basename(destination_path),
                        ) as progress_bar,
                    ):
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                    logger.debug(f"Downloaded {url} to {destination_path}")
                    return torch.load(destination_path)
                else:
                    buffer = io.BytesIO()
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            buffer.write(chunk)
                    buffer.seek(0)
                    logger.debug(f"Downloaded {url} to memory buffer")
                    return buffer.read()
            else:
                raise Exception(
                    f"Failed to download file: {url} with status code {response.status_code}"
                )
        except requests.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            raise

    def download(
        self,
        download_request: ModelDownloadRequest,
        destination_path: Optional[str] = None,
    ) -> bytes:
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
        return self._download_url(download_url, destination_path)


model = partial(RESTModel, model=Model)
model_version = partial(
    RESTModelVersion, model=ModelVersion, resource_path="model_version"
)
