
from langchain_core.documents import Document

from embedding_provider.embedding_provider import EmbeddingService  # Replace with actual import path
from chunker.document_chunker import DocumentChunker
from parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParser, DocumentProcessor

import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse, urljoin, unquote
from url.url_rules import URLRules
import hashlib


# Add the DocumentChunker class before the UrlSpider class
class UrlSpider(scrapy.Spider):
    name = "cssf_urls"
    start_urls = ["https://www.cssf.lu/en/"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = URLRules()
        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParser()])
        self.seen_hashes = set()
        # Initialize the DocumentChunker
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)

        # Initialize EmbeddingService with Milvus configuration
        milvus_config = {
            'host': 'localhost',  # Update with your Milvus host
            'port': '19530',  # Update with your Milvus port
            'collection_name': 'cssf_documents',  # Collection name for CSSF documents
            'connection_args': {"host": "localhost", "port": "19530"}
        }

        # Initialize embedding service (use_remote=True for SageMaker, False for local)
        self.embedding_service = EmbeddingService(
            use_remote=True,  # Set to False if you want to use local embeddings
            use_tei=True,  # Set to False if using legacy SageMaker handler
            milvus_config=milvus_config,
            endpoint_name='embedding-endpoint',  # Update with your SageMaker endpoint name
            region_name='eu-west-1'  # Update with your AWS region
        )

    def hash_document(self, doc: Document) -> str:
        base = doc.page_content + str(doc.metadata.get("source", ""))
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def parse(self, response):
        parsed_url = urlparse(response.url)
        domain = parsed_url.netloc

        if not self.rules.is_nested_only(response.url):
            elements = self.processor.process(response)

            # Use the DocumentChunker instead of manual chunking
            chunked_docs = self.chunker.chunk_document(elements, response.url)

            # Deduplication and Storage using EmbeddingService
            new_docs = []
            texts_to_store = []
            metadatas_to_store = []

            for doc in chunked_docs:
                doc_id = self.hash_document(doc)
                if doc_id in self.seen_hashes:
                    continue

                # Add doc_id to metadata
                doc.metadata["doc_id"] = doc_id
                self.seen_hashes.add(doc_id)
                new_docs.append(doc)

                # Prepare for batch storage
                texts_to_store.append(doc.page_content)
                metadatas_to_store.append(doc.metadata)

            # Store documents using EmbeddingService (batch operation)
            if new_docs:
                try:
                    result = self.embedding_service.add_texts_to_store(
                        texts=texts_to_store,
                        metadatas=metadatas_to_store
                    )
                    self.logger.info(f"Stored {result['count']} new documents from {response.url}")
                    self.logger.debug(f"Milvus IDs: {result['milvus_ids']}")
                except Exception as e:
                    self.logger.error(f"Failed to store documents from {response.url}: {str(e)}")

        if not self.rules.is_primary_domain(response.url):
            self.logger.info(f"No Primary URL: {response.url}")
            return

        content_type = response.headers.get("Content-Type", b"").decode("utf-8").split(";")[0]
        if not content_type.startswith("text/html"):
            return

        for href in response.css("a::attr(href)").getall():
            if not href:
                continue
            href = href.strip()
            full_url = urljoin(response.url, href)
            # full_url = canonicalize_url(full_url, keep_fragments=False)
            if self.rules.should_follow(full_url):
                self.rules.mark_visited(full_url)
                self.logger.info(f"Following primary: {full_url}")
                yield response.follow(full_url, callback=self.parse)


# === Run the spider ===
def run_spider(output_file="urls_raw.json"):
    process = CrawlerProcess(settings={
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 0.3,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36",
        "LOG_LEVEL": "INFO",
        "CLOSESPIDER_ITEMCOUNT": 0,
        # "CLOSESPIDER_PAGECOUNT": 50,
    })
    process.crawl(UrlSpider)
    process.start()

if __name__ == "__main__":
    run_spider()