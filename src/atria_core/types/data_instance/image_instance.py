from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.generic.ground_truth import GroundTruth
from atria_core.types.generic.image import Image


class ImageInstance(BaseDataInstance):  # type: ignore[misc]
    image: Image
    gt: GroundTruth = GroundTruth()
