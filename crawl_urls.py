import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse, urljoin, unquote
import json
import re
from w3lib.url import canonicalize_url
import os
import mimetypes
from parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParser, DocumentProcessor
from langchain_core.documents import Document

from unstructured.chunking.title import chunk_by_title
from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch

from url.url_rules import URLRules
import hashlib

from milvus_store import get_vectorstore

class UrlSpider(scrapy.Spider):
    name = "cssf_urls"
    start_urls = ["https://www.cssf.lu/en/"]

    vectorstore = get_vectorstore()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rules = URLRules()
        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParser()])
        self.seen_hashes = set()

    def hash_document(self, doc: Document) -> str:
        base = doc.page_content + str(doc.metadata.get("source", ""))
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def parse(self, response):
        parsed_url = urlparse(response.url)
        domain = parsed_url.netloc

        if not self.rules.is_nested_only(response.url):
            elements = self.processor.process(response)

            title_chunks = chunk_by_title(elements)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = []

            for chunk in title_chunks:
                if not isinstance(chunk.text, str) or not chunk.text.strip():
                    continue

                metadata = {
                    "source_url": response.url
                }
                split_docs.extend(splitter.create_documents([chunk.text], metadatas=[metadata]))

            # Deduplication and Storage
            new_docs = []
            for doc in split_docs:
                doc_id = self.hash_document(doc)
                if doc_id in self.seen_hashes:
                    continue
                doc.metadata["doc_id"] = doc_id
                self.seen_hashes.add(doc_id)
                new_docs.append(doc)

            if new_docs:
                self.vectorstore.add_documents(new_docs)
                self.logger.debug(f"Stored {len(new_docs)} new documents from {response.url}")

        if not self.rules.is_primary_domain(response.url):
            self.logger.info(f"No Primpary URL: {response.url}")
            return

        content_type = response.headers.get("Content-Type", b"").decode("utf-8").split(";")[0]
        if  not content_type.startswith("text/html"):
            return

        for href in response.css("a::attr(href)").getall():
            if not href:
                continue

            href = href.strip()

            full_url = urljoin(response.url, href)
            #full_url = canonicalize_url(full_url, keep_fragments=False)

            if  self.rules.should_follow(full_url):
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
        #"CLOSESPIDER_PAGECOUNT": 50,
    })

    process.crawl(UrlSpider)
    process.start()

if __name__ == "__main__":
    run_spider()