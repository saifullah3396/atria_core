from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.run import Run, RunCreate, RunUpdate


class RESTRun(RESTBase[Run, RunCreate, RunUpdate]):
    pass


run = partial(RESTRun, model=Run)
