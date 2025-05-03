from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.metric import Metric, MetricCreate, MetricUpdate


class RESTMetric(RESTBase[Metric, MetricCreate, MetricUpdate]):
    pass


metric = partial(RESTMetric, model=Metric)
