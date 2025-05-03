from functools import partial

from atria_core.logger import get_logger
from atria_core.rest.base import RESTBase
from atria_core.schemas.config import Config, ConfigCreate, ConfigUpdate

logger = get_logger(__name__)


class RESTConfig(RESTBase[Config, ConfigCreate, ConfigUpdate]):
    pass


config = partial(RESTConfig, model=Config)
