import torch

from atria_core.types.data_instance.document import DocumentInstance
from atria_core.types.generic.annotated_object import AnnotatedObject
from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.ground_truth import LayoutAnalysisGT
from atria_core.types.generic.image import Image
from atria_core.types.generic.label import Label

image = DocumentInstance(
    image=Image(content=torch.randn(3, 224, 224)),
)
print(image)

print(image.batched([image, image]))

ann1 = AnnotatedObject(
    bbox=BoundingBox(value=[0, 0, 224, 224]),
    label=Label(value=1, name="test1"),
)
ann2 = AnnotatedObject(
    bbox=BoundingBox(value=[0, 0, 224, 224]),
    label=Label(value=2, name="test2"),
)
ann3 = AnnotatedObject(
    bbox=BoundingBox(value=[0, 0, 224, 224]),
    label=Label(value=3, name="test3"),
)
gt1 = LayoutAnalysisGT(
    objects=[ann1, ann2, ann3],
)
ann1 = AnnotatedObject(
    bbox=BoundingBox(value=[0, 0, 36, 36]),
    label=Label(value=4, name="test4"),
    iscrowd=True,
)
ann2 = AnnotatedObject(
    bbox=BoundingBox(value=[0, 0, 36, 36]),
    label=Label(value=5, name="test5"),
)
ann3 = AnnotatedObject(
    bbox=BoundingBox(value=[0, 0, 36, 36]),
    label=Label(value=6, name="test6"),
    iscrowd=True,
)
gt2 = LayoutAnalysisGT(
    objects=[ann1, ann2, ann3],
)
print(gt1.batched([gt1, gt2]))
