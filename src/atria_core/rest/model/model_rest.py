from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.model import (
    Model,
    ModelCreate,
    ModelUpdate,
    ModelVersion,
    ModelVersionCreate,
    ModelVersionUpdate,
)


class RESTModel(RESTBase[Model, ModelCreate, ModelUpdate]):
    pass


class RESTModelVersion(RESTBase[ModelVersion, ModelVersionCreate, ModelVersionUpdate]):
    pass


model = partial(RESTModel, model=Model)
model_version = partial(RESTModelVersion, model=ModelVersion)
