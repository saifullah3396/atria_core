import json
from functools import partial

from omegaconf import OmegaConf

from atria_core.rest.base import RESTBase
from atria_core.schemas.base import DataInstanceType
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
from atria_core.types.datasets.splits import DatasetSplitType


class RESTDataset(RESTBase[Dataset, DatasetCreate, DatasetUpdate]):
    def upload_shard(
        self,
        is_public: bool,
        data_instance_type: DataInstanceType,
        config: dict,
        metadata: dict,
        dataset_split_type: DatasetSplitType,
        shard_index: int,
        total_shard_count: int,
        files: list,
    ) -> None:
        """Upload a file to the dataset."""
        response = self.client.post(
            self._url("upload_shard"),
            data={
                "is_public": is_public,
                "data_instance_type": data_instance_type.value,
                "config": json.dumps(OmegaConf.to_container(config)),
                "metadata": metadata.model_dump_json(),
                "dataset_split_type": dataset_split_type.value,
                "shard_index": shard_index,
                "total_shard_count": total_shard_count,
                "shard_index": shard_index,
            },
            files=files,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to upload shard {shard_index}: {response.status_code} - {response.text}"
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
    pass


class RESTShardFile(RESTBase[ShardFile, ShardFileCreate, ShardFileUpdate]):
    pass


dataset = partial(RESTDataset, model=Dataset)
dataset_split = partial(RESTDatasetSplit, model=DatasetSplit)
shard_file = partial(RESTShardFile, model=ShardFile)
