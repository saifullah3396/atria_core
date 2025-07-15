from faker import Faker

import factory
from atria_core.types.common import OCRType
from atria_core.types.data_instance.document_instance import DocumentInstance
from atria_core.types.data_instance.image_instance import ImageInstance
from atria_core.types.generic.annotated_object import (
    AnnotatedObject,
    AnnotatedObjectList,
)
from atria_core.types.generic.bounding_box import BoundingBox, BoundingBoxList
from atria_core.types.generic.ground_truth import (
    OCRGT,
    SERGT,
    ClassificationGT,
    GroundTruth,
    LayoutAnalysisGT,
    QuestionAnswerGT,
    VisualQuestionAnswerGT,
)
from atria_core.types.generic.image import Image
from atria_core.types.generic.label import Label, LabelList
from atria_core.types.generic.ocr import OCR
from atria_core.types.generic.question_answer_pair import (
    QuestionAnswerPair,
)

MOCK_HOCR_TESSERACT = """
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <title>HOCR Example - Tesseract Format</title>
</head>
<body>
    <div class='ocr_page' id='page_1' title='image "test_image.png"; bbox 0 0 800 600; ppageno 0'>
        <div class='ocr_carea' id='block_1_1' title='bbox 10 10 190 60'>
            <p class='ocr_par' id='par_1_1' title='bbox 10 10 190 60'>
                <span class='ocr_line' id='line_1_1' title='bbox 10 10 190 30'>
                    <span class='ocrx_word' id='word_1_1' title='bbox 10 10 60 30; x_wconf 95'>Sample</span>
                    <span class='ocrx_word' id='word_1_2' title='bbox 70 10 130 30; x_wconf 90'>HOCR</span>
                    <span class='ocrx_word' id='word_1_3' title='bbox 140 10 190 30; x_wconf 85'>Test</span>
                </span>
                <span class='ocr_line' id='line_1_2' title='bbox 10 35 190 60'>
                    <span class='ocrx_word' id='word_1_4' title='bbox 10 35 80 60; x_wconf 80'>Line</span>
                    <span class='ocrx_word' id='word_1_5' title='bbox 90 35 190 60; x_wconf 75'>Two</span>
                </span>
            </p>
        </div>
        <div class='ocr_carea' id='block_1_2' title='bbox 10 70 190 120'>
            <p class='ocr_par' id='par_1_2' title='bbox 10 70 190 120'>
                <span class='ocr_line' id='line_1_3' title='bbox 10 70 190 100'>
                    <span class='ocrx_word' id='word_1_6' title='bbox 10 70 80 100; x_wconf 92'>Second</span>
                    <span class='ocrx_word' id='word_1_7' title='bbox 90 70 190 100; x_wconf 88'>Paragraph</span>
                </span>
            </p>
        </div>
    </div>
</body>
</html>
"""

fake = Faker()


class LabelFactory(factory.Factory):
    class Meta:
        model = Label

    name = factory.LazyFunction(lambda: fake.word())
    value = factory.LazyFunction(lambda: fake.random_int(0, 10))


class LabelListFactory(factory.Factory):
    class Meta:
        model = LabelList

    @classmethod
    def _create(cls, model_class: LabelList, *args, **kwargs):
        return model_class.from_list(LabelFactory.build_batch(10))

    @classmethod
    def _build(cls, model_class: LabelList, *args, **kwargs):
        return model_class.from_list(LabelFactory.build_batch(10))


class BoundingBoxFactory(factory.Factory):
    class Meta:
        model = BoundingBox

    value = factory.LazyFunction(
        lambda: [
            fake.random_int(0, 100),
            fake.random_int(0, 100),
            fake.random_int(101, 200),
            fake.random_int(101, 200),
        ]
    )


class BoundingBoxListFactory(factory.Factory):
    class Meta:
        model = BoundingBoxList

    @classmethod
    def _create(cls, model_class: BoundingBoxList, *args, **kwargs):
        return model_class.from_list(BoundingBoxFactory.build_batch(10))

    @classmethod
    def _build(cls, model_class: BoundingBoxList, *args, **kwargs):
        return model_class.from_list(BoundingBoxFactory.build_batch(10))


class QuestionAnswerPairFactory(factory.Factory):
    class Meta:
        model = QuestionAnswerPair

    id = factory.LazyFunction(lambda: fake.random_int(1, 1000))
    question_text = factory.LazyFunction(lambda: fake.sentence())
    answer_start = factory.LazyFunction(lambda: [fake.random_int(0, 50)])
    answer_end = factory.LazyFunction(lambda: [fake.random_int(51, 100)])
    answer_text = factory.LazyFunction(lambda: [fake.sentence()])


class AnnotatedObjectFactory(factory.Factory):
    class Meta:
        model = AnnotatedObject

    label = factory.SubFactory(LabelFactory)
    bbox = factory.SubFactory(BoundingBoxFactory)
    segmentation = factory.LazyFunction(
        lambda: (
            [
                [
                    fake.pyfloat(min_value=0.0, max_value=200.0),
                    fake.pyfloat(min_value=0.0, max_value=200.0),
                ]
                for _ in range(6)
            ]
        )
    )
    iscrowd = factory.LazyFunction(lambda: fake.boolean())


class AnnotatedObjectListFactory(factory.Factory):
    class Meta:
        model = AnnotatedObjectList

    @classmethod
    def _create(cls, model_class: AnnotatedObjectList, *args, **kwargs):
        return model_class.from_list(AnnotatedObjectFactory.build_batch(10))

    @classmethod
    def _build(cls, model_class: AnnotatedObjectList, *args, **kwargs):
        return model_class.from_list(AnnotatedObjectFactory.build_batch(10))


class OCRFactory(factory.Factory):
    class Meta:
        model = OCR

    file_path = factory.LazyFunction(lambda: fake.file_path())
    type = factory.LazyFunction(lambda: fake.random_element(OCRType))
    content = factory.LazyFunction(lambda: MOCK_HOCR_TESSERACT)


class ImageFactory(factory.Factory):
    class Meta:
        model = Image

    # These are internal-use only — not passed to model
    _backend = factory.LazyFunction(lambda: "pil_file")
    _image_size = factory.LazyFunction(lambda: (32, 32))

    @factory.lazy_attribute
    def content(self):
        import numpy as np
        from PIL import Image as PILImage

        if self._backend == "pil":
            return PILImage.new("RGB", self._image_size, color="white")

        elif self._backend == "numpy":
            return np.random.randint(
                0, 256, (self._image_size[1], self._image_size[0], 3), dtype=np.uint8
            )

        elif self._backend == "torch":
            import torch

            return torch.randn((3, self._image_size[1], self._image_size[0]))

        elif self._backend == "pil_file":
            return None

        raise ValueError(f"Unsupported backend: {self._backend}")

    @factory.lazy_attribute
    def file_path(self):
        import tempfile

        from PIL import Image as PILImage

        if self._backend == "pil_file":
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            PILImage.new("RGB", self._image_size, color="white").save(temp_file.name)
            temp_file.close()
            return temp_file.name
        return None

    @classmethod
    def _build(cls, model_class, *args, **kwargs):
        kwargs.pop("_backend", None)
        kwargs.pop("_image_size", None)
        return model_class(*args, **kwargs)

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        kwargs.pop("_backend", None)
        kwargs.pop("_image_size", None)
        return model_class(*args, **kwargs)


class GroundTruthFactory(factory.Factory):
    class Meta:
        model = GroundTruth

    classification = factory.LazyFunction(
        lambda: ClassificationGT(label=LabelFactory.build())
    )
    ser = factory.LazyFunction(
        lambda: SERGT(
            words=fake.words(nb=5),
            word_bboxes=BoundingBoxListFactory.build(),
            word_labels=LabelListFactory.build(),
            segment_level_bboxes=BoundingBoxListFactory.build(),
        )
    )
    ocr = factory.LazyFunction(
        lambda: OCRGT(
            words=fake.words(nb=10),
            word_bboxes=BoundingBoxListFactory.build(),
            word_confs=[fake.pyfloat(min_value=0.0, max_value=1.0) for _ in range(10)],
            word_angles=[
                fake.pyfloat(min_value=0.0, max_value=360.0) for _ in range(10)
            ],
        )
    )
    qa = factory.LazyFunction(
        lambda: QuestionAnswerGT(
            qa_pair=QuestionAnswerPairFactory.build(), words=fake.words(nb=8)
        )
    )
    vqa = factory.LazyFunction(
        lambda: VisualQuestionAnswerGT(
            qa_pair=QuestionAnswerPairFactory.build(),
            words=fake.words(nb=8),
            word_bboxes=BoundingBoxListFactory.build(),
            segment_level_bboxes=BoundingBoxListFactory.build(),
        )
    )
    layout = factory.LazyFunction(
        lambda: LayoutAnalysisGT(
            annotated_objects=AnnotatedObjectListFactory.build(),
            words=fake.words(nb=15),
            word_bboxes=BoundingBoxListFactory.build(),
        )
    )


class ImageInstanceFactory(factory.Factory):
    class Meta:
        model = ImageInstance

    index = factory.LazyFunction(lambda: fake.random_int(min=0, max=1000))
    sample_id = factory.LazyFunction(lambda: str(fake.uuid4()))
    image = factory.SubFactory(ImageFactory)
    gt = factory.SubFactory(GroundTruthFactory)


class DocumentInstanceFactory(factory.Factory):
    class Meta:
        model = DocumentInstance

    index = factory.LazyFunction(lambda: fake.random_int(min=0, max=1000))
    sample_id = factory.LazyFunction(lambda: str(fake.uuid4()))
    image = factory.SubFactory(ImageFactory)
    ocr = factory.SubFactory(OCRFactory)
    gt = factory.SubFactory(GroundTruthFactory)
