import json
import os
import uuid
from functools import partial
from typing import Any, List, Optional

import requests
import tqdm

from atria.hub.utilities import ShardFilesBuffer
from atria_core.logger.logger import get_logger
from atria_core.rest.base import RESTBase
from atria_core.schemas.dataset import (
    Dataset,
    DatasetCreate,
    DatasetDownloadRequest,
    DatasetDownloadResponse,
    DatasetSplit,
    DatasetSplitCreate,
    DatasetSplitUpdate,
    DatasetUpdate,
    ShardFile,
    ShardFileCreate,
    ShardFileUpdate,
)
from atria_core.types.datasets.metadata import DatasetShardInfo

logger = get_logger(__name__)


class RESTDataset(RESTBase[Dataset, DatasetCreate, DatasetUpdate]):
    def get_or_create(
        self,
        *,
        obj_in: DatasetCreate,
    ) -> Optional[Dataset]:
        response = self.client.get(
            self._url("filter"),
            params=self._serialize_filters(
                dict(name=obj_in.name, config_name=obj_in.config_name)
            ),
        )
        if response.status_code == 200 and response.json():
            return self.model.model_validate(response.json())
        elif response.status_code == 404:
            return self.create(obj_in=obj_in)
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def request_download(
        self,
        download_request: DatasetDownloadRequest,
    ) -> DatasetDownloadResponse:
        """Request a download URL for the dataset."""
        response = self.client.post(
            self._url("request_download"),
            json=download_request.model_dump(),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to request download: {response.status_code} - {response.text}"
            )
        return DatasetDownloadResponse(**response.json())

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


class RESTDatasetSplit(RESTBase[DatasetSplit, DatasetSplitCreate, DatasetSplitUpdate]):
    def get_or_create(
        self,
        *,
        obj_in: DatasetSplitCreate,
        **kwargs: Any,
    ) -> Optional[Dataset]:
        response = self.client.get(
            self._url("filter"),
            params=self._serialize_filters(dict(name=obj_in.name.value, **kwargs)),
        )
        if response.status_code == 200 and response.json():
            return self.model.model_validate(response.json())
        elif response.status_code == 404:
            return self.create(obj_in=obj_in)
        else:
            raise RuntimeError(
                f"Failed to create {self.resource_path}: {response.status_code} - {response.text}"
            )

    def upload_shards(
        self,
        shard_info_list: List[DatasetShardInfo],
        dataset_split_id: uuid.UUID,
    ):
        """Helper function to upload a single shard with all files together."""

        with ShardFilesBuffer(shard_info_list) as shard_file_buffer:
            response = self.client.post(
                self._url("upload"),
                data={
                    "dataset_split_id": str(dataset_split_id),
                    "metadata": json.dumps(
                        [
                            {
                                "filesize": shard_info.filesize,
                                "index": shard_info.shard,
                                "nsamples": shard_info.nsamples,
                            }
                            for shard_info in shard_info_list
                        ]
                    ),
                },
                files=shard_file_buffer.files,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to upload shard files: {response.status_code} - {response.text}"
                )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to upload shard files: {response.status_code} - {response.text}"
            )


class RESTShardFile(RESTBase[ShardFile, ShardFileCreate, ShardFileUpdate]):
    pass


dataset = partial(RESTDataset, model=Dataset)
dataset_split = partial(
    RESTDatasetSplit, model=DatasetSplit, resource_path="dataset_split"
)
shard_file = partial(RESTShardFile, model=ShardFile, resource_path="shard_file")
