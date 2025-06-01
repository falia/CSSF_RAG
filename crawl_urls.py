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

# === Buffer CONFIGURATION ===
CHUNK_SIZE = 100
OUTPUT_FILE = "urls_raw.jsonl"
COLLECTED_BUFFER = []
WRITTEN_COUNT = 0

class UrlSpider(scrapy.Spider):
    name = "cssf_urls"
    start_urls = ["https://www.cssf.lu/en/"]

    # === Domain configuration ===
    primary_domains = ["cssf.lu"]
    secondary_domains = [
        "eur-lex.europa.eu",
        "data.europa.eu",
        "ata.legilux.public.lu"
    ]
    allowed_domains = primary_domains + secondary_domains

    exclude_url_patterns = [
        r"^https://www\.cssf\.lu/en/search",
        r"^https://www\.cssf\.lu/en/warnings",
        r"^https://www\.cssf\.lu/wp-content/uploads/annuaire_et_adresses_electroniques_specifiques",
        r"^https://www\.cssf\.lu/en/\d{4}/\d{2}/development-of-the-balance-sheet",
        r"^https://careers\.cssf\.lu",
        r"^tel:",
        r"^mailto:",
        r"^javascript:",
        r".*\.(zip|xls|xlsx|doc|docx)$",
        r"/(bg|es|cs|da|de|et|el|fr|ga|hr|it|lv|lt|hu|mt|nl|pl|pt|ro|sk|sl|fi|sv)/",
        r"^https?://eur-lex\.europa\.eu(?!.*?/en/txt/\?)",
        r"^https?://eur-lex\.europa\.eu/search\.html",
        r"^https?://data\.europa\.eu(?!/eli)",
        r"^https://data\.legilux\.public\.lu(?!/eli)",
        r"^https://edesk\.apps\.cssf\.lu"
    ]

    nested_only_patterns = [
        r"^https://www\.cssf\.lu/en/document",
        r"^https://www\.cssf\.lu/en/publication-data/",
        r"^https://www\.cssf\.lu/en/regulatory-framework/",
        r"^https://www\.cssf\.lu/en/?$"
    ]

    visited = set()
    collected = []

    file_counter = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs("output_files", exist_ok=True)

    def is_excluded_url(self, url):
        for pattern in self.exclude_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                self.logger.debug(f"Excluded by pattern: {pattern} → {url}")
                return True
        return False

    def is_allowed_domain(self, url):
        return any(domain in urlparse(url).netloc for domain in self.allowed_domains)

    def is_nested_only_url(self, url):
        for pattern in self.nested_only_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    def get_domain_type(self, url):
        domain = urlparse(url).netloc.lower().lstrip("www.")

        for primary in self.primary_domains:
            if domain == primary or domain.endswith("." + primary):
                return "primary"

        for secondary in self.secondary_domains:
            if domain == secondary or domain.endswith("." + secondary):
                return "secondary"

        return "unknown"

    def parse(self, response):
        global COLLECTED_BUFFER, WRITTEN_COUNT

        processor = DocumentProcessor(parsers=[CSSFHTMLParser(), PDFParser()])

        if not self.is_nested_only_url(response.url):
            # self.collected.append({"url": full_url})

            content_type = response.headers.get("Content-Type", b"").decode("utf-8").split(";")[0]
            elements = processor.process(response)

            yield {
                "url": response.url,
                "content_type": content_type,
                # "elements": [el.to_dict() for el in elements],
            }

            COLLECTED_BUFFER.append({
                "url": response.url,
                "content_type": content_type,
                # "elements": [el.to_dict() for el in elements],
            })
            WRITTEN_COUNT += 1
            if len(COLLECTED_BUFFER) >= CHUNK_SIZE:
                self.flush_buffer()

        for href in response.css("a::attr(href)").getall():
            if not href:
                continue

            href = href.strip()

            full_url = urljoin(response.url, href)

            if self.is_excluded_url(full_url):
                continue

            full_url = canonicalize_url(full_url, keep_fragments=False)

            if full_url in self.visited:
                continue

            if not self.is_allowed_domain(full_url):
                continue

            self.visited.add(full_url)

            if self.get_domain_type(full_url) == "primary" and response.headers.get("Content-Type", b"").startswith(b"text/html"):
                self.logger.info(f"Following primary: {full_url}")
                yield response.follow(full_url, callback=self.parse)

    def flush_buffer(self):
        global COLLECTED_BUFFER
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for entry in COLLECTED_BUFFER:
                json.dump(entry, f)
                f.write("\n")
        self.logger.info(f"Flushed {len(COLLECTED_BUFFER)} URLs to {OUTPUT_FILE}")
        COLLECTED_BUFFER.clear()

    def closed(self, reason):
        if COLLECTED_BUFFER:
            self.logger.info("Final buffer flush on spider close.")
            self.flush_buffer()

# === Run the spider ===
def run_spider(output_file="urls_raw.json"):
    process = CrawlerProcess(settings={
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 0.3,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36",
        "LOG_LEVEL": "INFO",
        "CLOSESPIDER_ITEMCOUNT": 0,
        "FEEDS": {
            output_file: {
                "format": "json",
                "encoding": "utf8",
                "indent": 2,
            }
        }
    })

    process.crawl(UrlSpider)
    process.start()

    print(f"\nTotal URLs written: {WRITTEN_COUNT}")
    print(f"Output file: {OUTPUT_FILE}\n")

if __name__ == "__main__":
    run_spider()
