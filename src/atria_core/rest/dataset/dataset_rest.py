import uuid
from functools import partial
from typing import Any, Optional

from atria.hub.utilities import ShardFileBuffer
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


class RESTShardFile(RESTBase[ShardFile, ShardFileCreate, ShardFileUpdate]):
    def upload(
        self,
        shard_info: DatasetShardInfo,
        dataset_split_id: uuid.UUID,
        is_last: bool = False,
    ):
        """Helper function to upload a single shard with all files together."""
        with ShardFileBuffer(shard_info) as shard_file_buffer:
            response = self.client.post(
                self._url("upload"),
                data={
                    "index": shard_info.shard,
                    "nsamples": shard_info.nsamples,
                    "filesize": shard_info.filesize,
                    "dataset_split_id": str(dataset_split_id),
                    "is_last": str(is_last),
                },
                files=shard_file_buffer.files,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to upload shard {shard_info.shard}: {response.status_code} - {response.text}"
                )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to upload shard {shard_info.shard}: {response.status_code} - {response.text}"
            )


dataset = partial(RESTDataset, model=Dataset)
dataset_split = partial(
    RESTDatasetSplit, model=DatasetSplit, resource_path="dataset_split"
)
shard_file = partial(RESTShardFile, model=ShardFile, resource_path="shard_file")
