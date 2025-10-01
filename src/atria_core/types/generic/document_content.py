from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.generic.bounding_box import BoundingBoxList
from atria_core.types.typing.common import OptListFloatField, OptListStrField


class DocumentContent(BaseDataModel):
    words: OptListStrField = None
    word_bboxes: BoundingBoxList | None = None
    word_segment_level_bboxes: BoundingBoxList | None = None
    word_confs: OptListFloatField = None
    word_angles: OptListFloatField = None
