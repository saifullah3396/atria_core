import bs4
import networkx
import tqdm

from atria_core.types.generic.bounding_box import BoundingBox
from atria_core.types.generic.ocr import OCRGraphNode, OCRLevel


class HOCRGraphParser:
    def __init__(self, ocr: str):
        self._soup = bs4.BeautifulSoup(ocr, features="xml")
        self._image_size = self._get_image_size()

    def _get_image_size(self) -> tuple[int, int]:
        page = self._soup.find("div", {"class": "ocr_page"})
        if page and "title" in page.attrs:
            bbox = page["title"].split("bbox")[1].split(";")[0].strip()
            x1, y1, x2, y2 = map(int, bbox.split())
            return x2, y2
        return (1, 1)

    def _add_node(
        self,
        graph: networkx.DiGraph,
        node_id: int,
        tag: bs4.Tag,
        level: str,
        parent_id: str | None = None,
    ):
        title = tag.get("title", "")
        bbox = None
        conf = None
        angle = 0.0

        # Parse bbox
        if "bbox" in title:
            bbox_str = title.split("bbox")[1].split(";")[0].strip()
            x1, y1, x2, y2 = map(int, bbox_str.split())
            w, h = self._image_size
            bbox = BoundingBox(value=[x1 / w, y1 / h, x2 / w, y2 / h])

        # Parse confidence
        if "x_wconf" in title:
            conf = float(title.split("x_wconf")[1].strip().split()[0])

        # Parse angle if available
        if "textangle" in title:
            angle = float(title.split("textangle")[1].split(";")[0])

        # Build node
        text = tag.get_text(strip=True)
        graph.add_node(
            node_id,
            **OCRGraphNode(
                id=node_id,
                word=text if level == OCRLevel.WORD.value else None,
                level=OCRLevel(level),
                bbox=bbox,
                conf=conf,
                angle=angle,
            ).model_dump(),
        )

        if parent_id:
            graph.add_edge(parent_id, node_id, relation="child")

        return node_id

    def parse_graph(self) -> dict:
        """
        Efficiently parse HOCR into a NetworkX directed graph using predictable structure.
        Returns:
            dict: Graph data in node-link format.
        """
        import networkx as nx
        from networkx.readwrite import json_graph

        graph = nx.DiGraph()
        node_id = 0

        add_node = self._add_node  # avoid attribute lookups

        pbar = tqdm.tqdm()
        for page_tag in self._soup.select("div.ocr_page"):
            page_id = add_node(graph, node_id, page_tag, "page")
            node_id += 1
            pbar.update(1)

            for block in page_tag.select("div.ocr_carea"):
                block_id = add_node(graph, node_id, block, "block", parent_id=page_id)
                node_id += 1
                pbar.update(1)

                for par in block.select("p.ocr_par"):
                    par_id = add_node(
                        graph, node_id, par, "paragraph", parent_id=block_id
                    )
                    node_id += 1
                    pbar.update(1)

                    for line in par.select("span.ocr_line"):
                        line_id = add_node(
                            graph, node_id, line, "line", parent_id=par_id
                        )
                        node_id += 1
                        pbar.update(1)

                        for word in line.select("span.ocrx_word"):
                            add_node(graph, node_id, word, "word", parent_id=line_id)
                            node_id += 1
                            pbar.update(1)

        return json_graph.node_link_data(graph)
