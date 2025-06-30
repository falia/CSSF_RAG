# run_cssf_spider.py
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from spiders.cssf_spider import CSSFSpider  # Adjust if your module path differs

def run_spider(mode="sitemap"):
    process = CrawlerProcess(settings={
        **get_project_settings(),
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 0.1,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36",
        "LOG_LEVEL": "INFO",
    })
    process.crawl(CSSFSpider, mode=mode)
    process.start()

if __name__ == "__main__":
    run_spider()
