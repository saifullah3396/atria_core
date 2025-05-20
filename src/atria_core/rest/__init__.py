from .config_rest import config
from .data_instances.document_instance_rest import document_instance
from .data_instances.image_instance_rest import image_instance
from .dataset.dataset_rest import dataset, dataset_split, shard_file
from .model.model_rest import model
from .tracking.experiment_rest import experiment
from .tracking.metric_rest import metric
from .tracking.param_rest import param
from .tracking.run_rest import run

# For a new basic set of rest operations you could just do
__all__ = [
    # tracking
    "experiment",
    "run",
    "param",
    "metric",
    # config
    "config",
    # model
    "model",
    # dataset
    "dataset",
    "dataset_split",
    "shard_file",
    # data instances
    "document_instance",
    "image_instance",
]
