import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse, urljoin, unquote
import json
import re
from w3lib.url import canonicalize_url
import os
import mimetypes
from parsers.parser import CSSFHTMLParser, PDFParser, DocumentProcessor
from langchain_core.documents import Document

from unstructured.chunking.title import chunk_by_title
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from pymilvus import connections
import torch

from url.url_rules import URLRules


class UrlSpider(scrapy.Spider):
    name = "cssf_urls"
    start_urls = ["https://www.cssf.lu/en/"]

    connections.connect("default", host="localhost", port="19530")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rules = URLRules()
        self.processor = DocumentProcessor(parsers=[CSSFHTMLParser(), PDFParser()])

    def parse(self, response):

        content_type = response.headers.get("Content-Type", b"").decode("utf-8").split(";")[0]

        if not self.rules.is_nested_only(response.url):
            elements = self.processor.process(response)

            title_chunks = chunk_by_title(elements)



        if  not content_type.startswith("text/html"):
            return

        for href in response.css("a::attr(href)").getall():
            if not href:
                continue

            href = href.strip()

            full_url = urljoin(response.url, href)
            full_url = canonicalize_url(full_url, keep_fragments=False)

            self.rules.mark_visited(full_url)

            if  self.rules.should_follow(full_url) :
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
    })

    process.crawl(UrlSpider)
    process.start()

if __name__ == "__main__":
    run_spider()
