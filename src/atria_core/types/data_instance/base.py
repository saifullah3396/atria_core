import uuid

from pydantic import Field

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import OptIntField, StrField


class BaseDataInstance(BaseDataModel):
    _tensor_data_model = "atria_core.types.data_instance._tensor.base.BaseDataInstance"
    index: OptIntField = None
    sample_id: StrField = Field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def key(self) -> str:
        """
        Generates a unique key for the data instance.

        The key is a combination of the UUID and the index (if present).

        Returns:
            str: The unique key for the data instance.
        """
        return str(self.sample_id.replace(".", "_"))
