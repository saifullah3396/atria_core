from functools import cached_property
import bs4
from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.ground_truth import OCRGT
from atria_core.types.generic.ocr import OCRType


class OCRProcessor:
    """
    OCRProcessor dynamically selects the appropriate parser based on the OCR type
    and directly calls the static parse() method of the underlying parser class.
    """

    @staticmethod
    def parse(raw_ocr: str, ocr_type: OCRType):
        ocr_type = ocr_type.lower()
        if ocr_type == OCRType.TESSERACT:
            return HOCRProcessor.parse(raw_ocr)
        # Add more parsers here as needed
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")


class HOCRProcessor:
    """
    HOCRProcessor parses HOCR (HTML-based OCR) data and extracts relevant information.
    """

    @staticmethod
    def parse(raw_ocr: str):
        soup = bs4.BeautifulSoup(raw_ocr, features="xml")

        # Extract image size
        pages = soup.findAll("div", {"class": "ocr_page"})
        image_size_str = pages[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())

        # Extract words and their properties
        words = []
        word_bboxes = []
        word_angles = []
        word_confs = []
        ocr_words = soup.findAll("span", {"class": "ocrx_word"})
        for word in ocr_words:
            title = word["title"]
            conf = int(title[title.find(";") + 10 :])
            if word.text.strip() == "":
                continue

            # Get text angle from line title
            textangle = 0
            parent_title = word.parent["title"]
            if "textangle" in parent_title:
                textangle = int(parent_title.split("textangle")[1][1:3])

            x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            words.append(word.text.strip())
            word_bboxes.append(BoundingBox(value=[x1 / w, y1 / h, x2 / w, y2 / h]))
            word_angles.append(textangle)
            word_confs.append(conf)

        return OCRGT(
            words=words,
            word_bboxes=word_bboxes,
            word_angles=word_angles,
            word_confs=word_confs,
        )
