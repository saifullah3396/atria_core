import json
from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.base import DataInstanceType
from atria_core.schemas.dataset import (
    Dataset,
    DatasetCreate,
    DatasetSplit,
    DatasetSplitCreate,
    DatasetSplitUpdate,
    DatasetUpdate,
    DatasetVersion,
    DatasetVersionCreate,
    DatasetVersionUpdate,
    ShardFile,
    ShardFileCreate,
    ShardFileUpdate,
)
from atria_core.types.datasets.splits import DatasetSplitType


class RESTDataset(RESTBase[Dataset, DatasetCreate, DatasetUpdate]):
    def upload_shard(
        self,
        name: str,
        is_public: bool,
        data_instance_type: DataInstanceType,
        version_tag: str,
        config: dict,
        metadata: dict,
        dataset_split_type: DatasetSplitType,
        shard_index: int,
        total_shard_count: int,
        files: list,
    ) -> None:
        """Upload a file to the dataset."""
        response = self.client.post(
            "/rest/v1/dataset/upload",
            data={
                "name": name,
                "is_public": is_public,
                "data_instance_type": data_instance_type.value,
                "version_tag": version_tag,
                "config": json.dumps(config),
                "metadata": json.dumps(metadata),
                "dataset_split_type": dataset_split_type.value,
                "shard_index": shard_index,
                "total_shard_count": total_shard_count,
                "shard_index": shard_index,
                "total_shard_count": total_shard_count,
            },
            files=files,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to upload shard {shard_index}: {response.status_code} - {response.text}"
            )


class RESTDatasetVersion(
    RESTBase[DatasetVersion, DatasetVersionCreate, DatasetVersionUpdate]
):
    pass


class RESTDatasetSplit(RESTBase[DatasetSplit, DatasetSplitCreate, DatasetSplitUpdate]):
    pass


class RESTShardFile(RESTBase[ShardFile, ShardFileCreate, ShardFileUpdate]):
    pass


dataset = partial(RESTDataset, model=Dataset)
dataset_version = partial(RESTDatasetVersion, model=DatasetVersion)
dataset_split = partial(RESTDatasetSplit, model=DatasetSplit)
shard_file = partial(RESTShardFile, model=ShardFile)
