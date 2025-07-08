from typing import TYPE_CHECKING

import torch

from atria_core.types.base.data_model import TensorDataModel
from atria_core.types.data_instance._tensor.base import TensorBaseDataInstance
from atria_core.types.generic._tensor.ground_truth import TensorGroundTruth
from atria_core.types.generic._tensor.image import TensorImage
from atria_core.types.generic._tensor.ocr import TensorOCR

if TYPE_CHECKING:
    from atria_core.types.data_instance._raw.document_instance import DocumentInstance  # noqa


class TensorDocumentInstance(  # type: ignore[misc]
    TensorBaseDataInstance, TensorDataModel["DocumentInstance"]
):
    _raw_model = (
        "atria_core.types.data_instance._raw.document_instance.DocumentInstance"
    )
    page_id: torch.Tensor
    total_num_pages: torch.Tensor
    image: TensorImage
    ocr: TensorOCR | None = None
    gt: TensorGroundTruth = TensorGroundTruth()
