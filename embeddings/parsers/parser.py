from abc import ABC, abstractmethod
import os
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
    def can_process(self, url: str, content_type: str = None) -> bool:
        pass

    @abstractmethod
    def parse(self, content: bytes, url: str, content_type: str):
        pass


class EurlexHTMLParser(DocumentParser):
    def can_process(self, url: str, content_type: str = None) -> bool:
        url_match = "eur-lex.europa.eu" in url and not url.lower().endswith(".pdf")
        content_type_match = content_type and "text/html" in content_type
        return url_match and content_type_match

    def parse(self, content: bytes, url: str, content_type: str):
        # Create a mock response to extract content using CSS selectors
        response = HtmlResponse(url=url, body=content, encoding='utf-8')

        raw_sections = response.css("div.PP4Contents").getall()
        raw_html = "\n\n".join(raw_sections)

        if raw_html:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(raw_html)
                temp_file_path = temp_file.name

            try:
                elements = partition_html(
                    filename=temp_file_path,

                    # HTML doesn't have strategies like PDF
                    # Remove: strategy="hi_res"

                    # HTML-specific parameters
                    include_page_breaks=False,  # HTML has no pages
                    infer_table_structure=True,  # Keep for tables
                    skip_headers_and_footers=True,  # Skip nav content

                    # Chunking (same as your PDF for consistency)
                    max_characters=1500,
                    combine_text_under_n_chars=300,
                    new_after_n_chars=1200,

                    chunking_strategy="by_title",

                    # Language support
                    languages=["eng", "fra"],
                )
                return elements
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        else:
            return []


class CSSFHTMLParser(DocumentParser):
    def can_process(self, url: str, content_type: str = None) -> bool:
        url_match = "www.cssf.lu" in url and not url.lower().endswith(".pdf")
        content_type_match = content_type and "text/html" in content_type
        return url_match and content_type_match

    def parse(self, content: bytes, url: str, content_type: str):
        # Create a mock response to extract content using CSS selectors
        response = HtmlResponse(url=url, body=content, encoding='utf-8')

        raw_sections = response.css("div.content-section").getall()
        raw_html = "\n\n".join(raw_sections)

        if raw_html:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(raw_html)
                temp_file_path = temp_file.name

            try:
                elements = partition_html(
                    filename=temp_file_path,

                    # HTML doesn't have strategies like PDF
                    # Remove: strategy="hi_res"

                    # HTML-specific parameters
                    include_page_breaks=False,  # HTML has no pages
                    infer_table_structure=True,  # Keep for tables
                    skip_headers_and_footers=True,  # Skip nav content

                    # Chunking (same as your PDF for consistency)
                    max_characters=1500,
                    combine_text_under_n_chars=300,
                    new_after_n_chars=1200,

                    extract_image_block_types=["Image", "Table"],
                    chunking_strategy="by_title",

                    # Language support
                    languages=["eng", "fra"],
                )
                return elements
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        else:
            return []


class PDFParser(DocumentParser):
    def can_process(self, url: str, content_type: str = None) -> bool:
        url_match = url.lower().endswith(".pdf")
        content_type_match = content_type and "application/pdf" in content_type
        return url_match or content_type_match  # PDF can use OR logic since it's more specific

    def parse(self, content: bytes, url: str, content_type: str):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        try:
            elements = partition_pdf(
                filename=temp_pdf_path,

                strategy="hi_res",
                infer_table_structure=True,
                include_page_breaks=True,
                languages=["eng", "fra"],

                max_characters=1500,
                combine_text_under_n_chars=300,
                new_after_n_chars=1200,

                hi_res_model_name = "yolox",
                #hi_res_model_name="detectron2_onnx",
                extract_images_in_pdf=False,
                extract_image_block_types=["Image", "Table"],
                chunking_strategy="by_title"
            )
            return elements
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_pdf_path)
            except:
                pass


class GenericHTMLParser(DocumentParser):
    """Fallback parser for HTML content that doesn't match specific parsers."""

    def can_process(self, url: str, content_type: str = None) -> bool:
        # Only use as fallback for HTML content
        return content_type and "text/html" in content_type

    def parse(self, content: bytes, url: str, content_type: str):
        # For generic HTML, just partition the entire content
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".html", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            elements = partition_html(
                filename=temp_file_path,

                # HTML doesn't have strategies like PDF
                # Remove: strategy="hi_res"

                # HTML-specific parameters
                include_page_breaks=False,  # HTML has no pages
                infer_table_structure=True,  # Keep for tables
                skip_headers_and_footers=True,  # Skip nav content

                # Chunking (same as your PDF for consistency)
                max_characters=1500,
                combine_text_under_n_chars=300,
                new_after_n_chars=1200,

                # Language support
                languages=["eng", "fra"],
                chunking_strategy="by_title"
            )
            return elements
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass


# --- Parser manager (or factory) ---
class DocumentProcessor:
    def __init__(self, parsers):
        self.parsers = parsers

    def process(self, content: bytes, url: str, content_type: str):
        """
        Process document content using appropriate parser.

        Args:
            content: Raw bytes of the document
            url: Original URL of the document
            content_type: MIME type of the document

        Returns:
            List of unstructured elements
        """
        for parser in self.parsers:
            if parser.can_process(url, content_type):
                logger.info(f"Using {parser.__class__.__name__} for {url}")
                return parser.parse(content, url, content_type)

        logger.warning(f"No parser available for URL: {url}, content-type: {content_type}")
        return []  # Return empty list if no parser can handle the content


# === Example usage ===
# === Example usage ===
if __name__ == "__main__":
    processor = DocumentProcessor(parsers=[
        EurlexHTMLParser(),
        CSSFHTMLParser(),
        PDFParser(),
        GenericHTMLParser()
    ])

    #url = "https://www.cssf.lu/wp-content/uploads/cssf25_892eng.pdf"
    url = "https://www.cssf.lu/en/2025/07/new-procedure-internalised-settlement-reporting-under-article-9-of-csdr-available-since-1-july-2025/"
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("Content-Type", "application/pdf")

        elements = processor.process(content=content, url=url, content_type=content_type)
        print(f"Loaded {len(elements)} sections.")

        for i, element in enumerate(elements):
            print(f"\n--- Element {i+1} ---")
            print(f"Type: {type(element)}")
            if hasattr(element, 'category'):
                print(f"Category: {element.category}")
            if hasattr(element, 'text'):
                print(f"Text: {element.text}")
            else:
                print(element)
    except Exception as e:
        logger.error(f"Failed to download or process the document: {e}")
    #     content_type="application/pdf"
    # )
    #
    # print(f"Loaded {len(elements)} sections.")