import io
import os
from functools import partial
from pathlib import Path
from typing import Optional
import uuid

import requests
import torch
import tqdm

from atria_core.logger.logger import get_logger
from atria_core.rest.base import RESTBase
from atria_core.schemas.model import Model, ModelCreate, ModelUpdate
from atria_core.types.tasks import TaskType

logger = get_logger(__name__)


class RESTModel(RESTBase[Model, ModelCreate, ModelUpdate]):
    def upload(
        self,
        registry_name: str,
        task: Optional[TaskType] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        is_public: bool = False,
        ckpt_path: Optional[str] = None,
        ckpt_state_path: Optional[str] = "state_dict",
        model_state_path: Optional[str] = "_model",
    ) -> None:
        files = {}
        model_file = open(ckpt_path, "rb") if ckpt_path else None
        if ckpt_path is not None:
            assert os.path.exists(
                ckpt_path
            ), f"Checkpoint path {ckpt_path} does not exist"
            assert os.path.isfile(
                ckpt_path
            ), f"Checkpoint path {ckpt_path} is not a file"
            files = {
                "model_file": (
                    "model.bin",
                    model_file,
                    "application/octet-stream",
                ),
            }
        try:
            response = self.client.post(
                self._url("upload"),
                data=dict(
                    registry_name=registry_name,
                    task=task,
                    model_type=model_type,
                    model_name=model_name,
                    description=description,
                    is_public=is_public,
                    ckpt_state_path=ckpt_state_path,
                    model_state_path=model_state_path,
                ),
                files=files,
            )
        finally:
            if model_file is not None:
                model_file.close()
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

    def load(
        self,
        model_name: Optional[str] = None,
        model_id: Optional[uuid.UUID] = None,
        destination_path: Optional[str] = None,
    ) -> bytes:
        try:
            assert (
                model_id is not None or model_name is not None
            ), "Either model_version_id or model_name must be provided."
            if model_id is not None:
                assert (
                    model_name is None
                ), "model_name and model_id are mutually exclusive"
                model = self.get(id=model_id)
            else:
                assert (
                    model_id is None
                ), "model_name and model_id are mutually exclusive"
                if "@" in model_name:
                    username, model_name = model_name.split("@")
                else:
                    username = None

                model = self.filter(
                    name=model_name,
                    username=username,
                )
            return self._download_url(model.model_uri, destination_path)
        except Exception as e:
            logger.error(f"Model {model_name} not found in the registry.")
            raise e


model = partial(RESTModel, model=Model)
