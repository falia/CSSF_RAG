from abc import ABC, abstractmethod
import os
#from langchain_unstructured import UnstructuredLoader
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from scrapy.http import HtmlResponse
import tempfile
import json
from unstructured.documents.elements import Element
import logging

logger = logging.getLogger(__name__)

class DocumentParser(ABC):
    @abstractmethod
    def can_process(self, url: str) -> bool:
        pass

    @abstractmethod
    def parse(self, response):
        pass

class CSSFHTMLParser(DocumentParser):
    def can_process(self, url: str) -> bool:
        return "www.cssf.lu" in url and not url.lower().endswith(".pdf")

    def parse(self, response):
        raw_sections = response.css("div.content-section").getall()
        raw_html = "\n\n".join(raw_sections)

        if raw_html:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(raw_html)
                temp_file_path = temp_file.name

            return partition_html(
                temp_file_path,
                mode="elements",
                unstructured_kwargs={"strategy": "hi_res"}
            )

        else: return []

class PDFParser(DocumentParser):
    def can_process(self, url: str) -> bool:
        return url.lower().endswith(".pdf")

    def parse(self, response):
        pdf_bytes = response.body

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        return partition_pdf(
            temp_pdf_path,
            mode="elements",
            unstructured_kwargs={"strategy": "hi_res"}
        )

# --- Parser manager (or factory) ---
class DocumentProcessor:
    def __init__(self, parsers):
        self.parsers = parsers

    def process(self, response):
        for parser in self.parsers:
            if parser.can_process(response.url):
                return parser.parse(response)
        
        logger.warning(f"No parser available for URL: {response.url}")
        return []  # Return empty list or None depending on expected downstream behavior

# === Example usage ===
#if __name__ == "__main__":
    #    file_path = "C:\\Users\\faton\\workspace\\tutorial\\output\\pdf\\document.pdf"  # or .html

    #    processor = DocumentProcessor(parsers=[HTMLParser(), PDFParser()])
    #    documents = processor.process(file_path)

    #    print(f"Loaded {len(documents)} sections.\n")
        #    for i, doc in enumerate(documents):
        #print(f"--- Document #{i+1} ---")
        #print("Metadata:", doc.metadata)
        #print("Content:\n", doc.page_content[:500])
        #print()
