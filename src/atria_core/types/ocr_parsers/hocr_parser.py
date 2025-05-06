from functools import cached_property

import bs4

from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.ground_truth import OCRGroundTruth


class HOCRProcessor:
    def __init__(self, ocr: str):
        self._soup = bs4.BeautifulSoup(ocr, features="xml")

    @cached_property
    def image_size(self):
        image_size_str = self.pages[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())
        return (w, h)

    @cached_property
    def pages(self):
        return self._soup.findAll("div", {"class": "ocr_page"})

    @cached_property
    def blocks(self):
        return self._soup.findAll("div", {"class": "ocr_carea"})

    @cached_property
    def words(self):
        return self._soup.findAll("span", {"class": "ocrx_word"})

    def pars_per_block(self, block):
        return block.findAll("p", {"class": "ocr_par"})

    def lines_per_par(self, par):
        return par.findAll("span", {"class": "ocr_line"})

    def words_per_line(self, line):
        return line.findAll("span", {"class": "ocrx_word"})

    def parse(self):
        words = []
        word_bboxes = []
        word_angles = []
        word_confs = []
        w, h = self.image_size
        for word in self.words:
            title = word["title"]
            conf = int(title[title.find(";") + 10 :])
            if word.text.strip() == "":
                continue

            # get text angle from line title
            textangle = 0
            parent_title = word.parent["title"]
            if "textangle" in parent_title:
                textangle = int(parent_title.split("textangle")[1][1:3])

            x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            words.append(word.text.strip())
            word_bboxes.append(BoundingBox(value=[x1 / w, y1 / h, x2 / w, y2 / h]))
            word_angles.append(textangle)
            word_confs.append(conf)
        return OCRGroundTruth(
            words=words,
            word_bboxes=word_bboxes,
            word_angles=word_angles,
            word_confs=word_confs,
        )
