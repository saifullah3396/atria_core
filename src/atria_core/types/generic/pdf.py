from functools import cached_property

from PIL.Image import Image as PILImage

from atria_core.types.base.data_model import BaseDataModel
from atria_core.types.typing.common import IntField, OptStrField


class PDF(BaseDataModel):
    file_path: OptStrField = None
    num_pages: IntField = None

    @cached_property
    def pages(self) -> list[PILImage]:
        """Lazily extract pages from PDF when accessed."""
        if self.file_path is None:
            raise ValueError(
                "PDF file path is not set. Please set file_path before accessing pages."
            )

        from pdf2image import convert_from_path

        return convert_from_path(self.file_path)

    def get_page(self, page_num: int) -> PILImage:
        """Get a specific page from the PDF (0-indexed)."""
        if page_num < 0 or (self.num_pages is not None and page_num >= self.num_pages):
            raise IndexError(
                f"Page {page_num} is out of range. PDF has {self.num_pages} pages."
            )

        return self.pages[page_num]

    def _load(self):
        """Load PDF metadata without extracting pages."""
        if self.file_path is None:
            raise ValueError(
                "PDF file path is not set. Please set file_path before loading."
            )

        # Load number of pages if not already set
        if self.num_pages is None:
            try:
                import PyPDF2

                with open(self.file_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    self.num_pages = len(pdf_reader.pages)
            except ImportError:
                # Fallback to pdf2image if PyPDF2 is not available
                from pdf2image import convert_from_path

                pages = convert_from_path(self.file_path)
                self.num_pages = len(pages)
