from functools import partial

from atria_core.rest.base import RESTBase
from atria_core.schemas.data_instances.document_instance import (
    DocumentInstance,
    DocumentInstanceCreate,
    DocumentInstanceUpdate,
)


class RESTDocumentInstance(
    RESTBase[DocumentInstance, DocumentInstanceCreate, DocumentInstanceUpdate]
):
    pass


document_instance = partial(
    RESTDocumentInstance, model=DocumentInstance, resource_path="document_instance"
)
