from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.param import Param, ParamCreate, ParamUpdate


class RESTParam(RESTBase[Param, ParamCreate, ParamUpdate]):
    pass


param = partial(RESTParam, model=Param)
