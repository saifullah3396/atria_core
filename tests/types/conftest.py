import tempfile

import numpy as np
import torch
from faker import Faker
from PIL import Image as PILImage
from polyfactory import Use
from polyfactory.factories.pydantic_factory import ModelFactory

from atria.data.structures.data_instance.document import DocumentInstance
from atria.data.structures.data_instance.image import ImageInstance
from atria.data.structures.data_instance.tokenized_object import TokenizedObjectInstance
from atria.data.structures.generic.annotated_object import (
    AnnotatedObject,
    AnnotatedObjectSequence,
)
from atria.data.structures.generic.bounding_box import (
    BoundingBox,
    BoundingBoxMode,
    BoundingBoxSequence,
)
from atria.data.structures.generic.image import Image
from atria.data.structures.generic.label import Label, LabelSequence
from atria.data.structures.generic.ocr import (
    OCR,
    OCRGraph,
    OCRGraphLink,
    OCRGraphNode,
    OCRType,
)
from atria.data.structures.generic.question_answer_pair import (
    QuestionAnswerPair,
    QuestionAnswerPairSequence,
)
from atria.data.structures.ocr_parsers.hocr_graph_parser import HOCRGraphParser

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


class LabelFactory(ModelFactory[Label]):
    __model__ = Label
    value = Use(lambda: torch.randint(0, 10, (1,)).item())


class LabelSequenceFactory(ModelFactory[LabelSequence]):
    __model__ = LabelSequence
    values = Use(lambda: torch.randint(0, 10, (16,)).tolist())
    names = Use(lambda: [f"label_{i}" for i in range(16)])


class BoundingBoxFactory(ModelFactory[BoundingBox]):
    __model__ = BoundingBox
    value = Use(lambda: torch.randint(0, 256, (4,)).tolist())


class SequenceBoundingBoxesFactory(ModelFactory[BoundingBoxSequence]):
    __model__ = BoundingBoxSequence
    value = Use(lambda: [torch.randint(0, 256, (4,)) for _ in range(16)])
    backend: str = "torch"

    @classmethod
    def value(cls) -> object:
        backend = cls.backend
        if backend == "np":
            return np.random.randint(0, 256, (16, 4))
        elif backend == "list":
            return [torch.randint(0, 256, (4,)) for _ in range(16)]
        elif backend == "torch":
            return torch.randint(0, 256, (16, 4))
        else:
            raise ValueError(f"Unsupported backend: {backend}")


class QuestionAnswerPairsFactory(ModelFactory[QuestionAnswerPair]):
    __model__ = QuestionAnswerPair


class QuestionAnswerPairSequenceFactory(ModelFactory[QuestionAnswerPairSequence]):
    __model__ = QuestionAnswerPairSequence


class AnnotatedObjectFactory(ModelFactory[AnnotatedObject]):
    __model__ = AnnotatedObject
    label = Use(lambda: LabelFactory.build())
    bbox = Use(lambda: BoundingBoxFactory.build(mode=BoundingBoxMode.XYXY))
    segmentation = Use(lambda: torch.randint(0, 256, (10, 10)).tolist())
    iscrowd = Use(lambda: False)


class AnnotatedObjectSequenceFactory(ModelFactory[AnnotatedObjectSequence]):
    __model__ = AnnotatedObjectSequence
    label = Use(lambda: LabelSequenceFactory.build())
    bbox = Use(lambda: SequenceBoundingBoxesFactory.build(mode=BoundingBoxMode.XYXY))
    segmentation = Use(lambda: torch.randint(0, 256, (16, 10, 10)))
    iscrowd = Use(lambda: torch.ones(16, dtype=torch.bool).tolist())


class OCRGraphNodeFactory(ModelFactory[OCRGraphNode]):
    __model__ = OCRGraphNode
    id = Use(lambda: Faker().random_int(min=0, max=100))
    word = Use(lambda: Faker().word())
    level = Use(lambda: Faker().random_element(elements=("word", "line", "paragraph")))
    bbox = Use(lambda: BoundingBoxFactory.build(mode=BoundingBoxMode.XYXY))
    conf = Use(lambda: round(Faker().pyfloat(min_value=0, max_value=1), 2))
    angle = Use(lambda: round(Faker().pyfloat(min_value=0, max_value=360), 2))
    label = Use(lambda: LabelFactory.build())


class OCRGraphLinkFactory(ModelFactory[OCRGraphLink]):
    __model__ = OCRGraphLink
    source = Use(lambda: Faker().random_int(min=0, max=100))
    target = Use(lambda: Faker().random_int(min=0, max=100))
    relation = Use(
        lambda: Faker().random_element(elements=("parent", "child", "sibling"))
    )


class OCRGraphFactory(ModelFactory[OCRGraph]):
    __model__ = OCRGraph
    directed = Use(lambda: Faker().boolean())
    multigraph = Use(lambda: Faker().boolean())
    graph = Use(lambda: {"name": Faker().word()})
    nodes = Use(lambda: [OCRGraphNodeFactory.build() for _ in range(5)])
    links = Use(lambda: [OCRGraphLinkFactory.build() for _ in range(3)])


class OCRFactory(ModelFactory[OCR]):
    __model__ = OCR
    file_path = Use(lambda: Faker().file_path())
    ocr_type = Use(lambda: OCRType.TESSERACT)
    graph = Use(lambda: OCRGraphFactory.build())
    backend: str = "from_file"

    @classmethod
    def graph(cls) -> object:
        if cls.backend == "from_factory":
            return OCRGraphFactory.build()
        elif cls.backend == "from_file":
            return HOCRGraphParser(MOCK_HOCR_TESSERACT).parse()
        else:
            raise ValueError(f"Unsupported backend: {cls.backend}")


class ImageFactory(ModelFactory[Image]):
    __model__ = Image

    backend: str = "torch"
    image_size: tuple[int, int] = (256, 256)

    @classmethod
    def content(cls) -> object:
        if cls.backend == "pil":
            return PILImage.new("RGB", cls.image_size, color="white")

        elif cls.backend == "numpy":
            return np.random.randint(
                0, 256, (cls.image_size[1], cls.image_size[0], 3), dtype=np.uint8
            )

        elif cls.backend == "torch":
            return torch.randn((3, cls.image_size[1], cls.image_size[0]))

        elif cls.backend == "pil_file":
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            PILImage.new("RGB", cls.image_size, color="white").save(temp_file.name)
            temp_file.close()
            return None

    @classmethod
    def file_path(cls) -> str:
        if cls.backend == "pil_file":
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            PILImage.new("RGB", cls.image_size, color="white").save(temp_file.name)
            temp_file.close()
            return temp_file.name
        return None


class ImageInstanceFactory(ModelFactory[ImageInstance]):
    __model__ = ImageInstance
    index = Use(lambda: 0)
    image = Use(lambda: ImageFactory.build())
    label = Use(lambda: LabelFactory.build())


class DocumentInstanceFactory(ModelFactory[DocumentInstance]):
    __model__ = DocumentInstance
    index = Use(lambda: 0)
    image = Use(lambda: ImageFactory.build())
    ocr = Use(lambda: OCRFactory.build())
    label = Use(lambda: LabelFactory.build())
    question_answer_pairs = Use(lambda: QuestionAnswerPairSequenceFactory.build())
    annotated_objects = Use(lambda: AnnotatedObjectSequenceFactory.build())


class TokenizedObjectInstanceFactory(ModelFactory[TokenizedObjectInstance]):
    __model__ = TokenizedObjectInstance
    index = Use(lambda: 0)
    token_ids = Use(lambda: torch.randn(16))
    word_ids = Use(lambda: torch.randn(16))
    token_labels = Use(lambda: torch.randn(16))
    token_type_ids = Use(lambda: torch.randn(16))
    attention_mask = Use(lambda: torch.randn(16))
    token_bboxes = Use(lambda: torch.randn(16, 4))
    sequence_ids = Use(lambda: torch.randn(16))
    image = Use(lambda: ImageFactory.build())
    label = Use(lambda: LabelFactory.build())
