import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse, urljoin
import json
import re
from w3lib.url import canonicalize_url


# === Buffer CONFIGURATION ===
CHUNK_SIZE = 1000
OUTPUT_FILE = "urls_raw.jsonl"
COLLECTED_BUFFER = []
WRITTEN_COUNT = 0

class UrlSpider(scrapy.Spider):
    name = "cssf_urls"
    start_urls = ["https://www.cssf.lu/en/"]

    allowed_domains = [
        "cssf.lu",
        "eur-lex.europa.eu",
        "data.europa.eu",
        "ata.legilux.public.lu"
    ]

    exclude_url_patterns = [
        #r"^https://www\.cssf\.lu/en/document",
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
        r"^https://data\.legilux\.public\.lu(?!/eli)"
    ]

    visited = set()
    collected = []

    #def is_excluded_url(self, url):
        #return any(re.match(pattern, url, re.IGNORECASE) for pattern in self.exclude_url_patterns)

    def is_excluded_url(self, url):
        for pattern in self.exclude_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                #self.logger.info(f"Excluded by pattern: {pattern} → {url}")
                return True
        return False

    def is_allowed_domain(self, url):
        return any(domain in urlparse(url).netloc for domain in self.allowed_domains)

    #def normalize_url(self, href: str, base_url: str) -> str:
        #parsed = urlparse(href)
        #return

    def normalize_url(self, href: str, base_url: str) -> str:
        resolved = urljoin(base_url, href)
        return canonicalize_url(resolved, keep_fragments=False)

    def parse(self, response):
        global COLLECTED_BUFFER, WRITTEN_COUNT

        if not response.headers.get("Content-Type", b"").startswith(b"text/html"):
            self.logger.info(f"Skipping non-HTML content: {response.url}")
            return

        for href in response.css("a::attr(href)").getall():
            if not href:
                continue

            full_url = urljoin(response.url, href)

            if self.is_excluded_url(full_url):
                continue

            full_url = canonicalize_url(full_url, keep_fragments=False)

            #full_url = self.normalize_url(href, response.url)

            if full_url in self.visited:
                continue

            if not self.is_allowed_domain(full_url):
                continue

            self.visited.add(full_url)
            self.collected.append({"url": full_url})
            yield {"url": full_url}


            COLLECTED_BUFFER.append({"url": full_url})
            WRITTEN_COUNT += 1
            if len(COLLECTED_BUFFER) >= CHUNK_SIZE:
                self.flush_buffer()

            self.logger.info(f"Following: {full_url}")
            yield response.follow(full_url, callback=self.parse)

    def flush_buffer(self):
        global COLLECTED_BUFFER
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for entry in COLLECTED_BUFFER:
                json.dump(entry, f)
                f.write("\n")
        self.logger.info(f"Flushed {len(COLLECTED_BUFFER)} URLs to {OUTPUT_FILE}")
        COLLECTED_BUFFER.clear()

# Run spider and save output
def run_spider(output_file="urls_raw.json"):
    process = CrawlerProcess(settings={
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 16,
        "DOWNLOAD_DELAY": 0.5,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36",
        "LOG_LEVEL": "INFO",
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

    #print(f"\nSaved {len(spider.collected)} URLs to {output_file}\n")
    if COLLECTED_BUFFER:
        print("Flushing remaining URLs...")
        spider.flush_buffer()

    print(f"\nTotal URLs written: {WRITTEN_COUNT}")
    print(f"Output file: {OUTPUT_FILE}\n")

if __name__ == "__main__":
    run_spider()
