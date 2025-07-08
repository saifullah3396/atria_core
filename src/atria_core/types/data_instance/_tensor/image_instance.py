from typing import TYPE_CHECKING

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.data_instance._tensor.base import TensorBaseDataInstance
from atria_core.types.generic._tensor.ground_truth import TensorGroundTruth
from atria_core.types.generic._tensor.image import TensorImage

if TYPE_CHECKING:
    from atria_core.types.data_instance._raw.image_instance import ImageInstance  # noqa


class TensorImageInstance(TensorBaseDataInstance, TensorDataModel["ImageInstance"]):  # type: ignore[misc]
    _raw_model = "atria_core.types.data_instance._raw.image_instance.ImageInstance"
    image: TensorImage
    gt: TensorGroundTruth = TensorGroundTruth()
