from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.experiment import Experiment, ExperimentCreate, ExperimentUpdate


class RESTExperiment(RESTBase[Experiment, ExperimentCreate, ExperimentUpdate]):
    pass


experiment = partial(RESTExperiment, model=Experiment)
