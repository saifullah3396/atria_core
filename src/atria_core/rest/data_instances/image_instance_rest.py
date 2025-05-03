from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.data_instances.image_instance import (
    ImageInstance,
    ImageInstanceCreate,
    ImageInstanceUpdate,
)


class RESTImageInstance(
    RESTBase[ImageInstance, ImageInstanceCreate, ImageInstanceUpdate]
):
    pass


image_instance = partial(RESTImageInstance, model=ImageInstance)
