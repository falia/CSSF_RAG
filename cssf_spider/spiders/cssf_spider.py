import os
import json
import boto3
import scrapy
from datetime import datetime
from urllib.parse import urlparse
from utils.sitemap_helper import SitemapHelper
from utils.metadata_extractor import MetadataExtractor
import pytz
import requests


def safe_filename_from_url(url: str) -> str:
    from base64 import urlsafe_b64encode
    return urlsafe_b64encode(url.encode("utf-8")).decode("ascii")


class CSSFSpider(scrapy.Spider):
    name = "cssf_spider"
    allowed_domains = ["cssf.lu"]

    def __init__(self, mode="sitemap", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.visited = set()
        self.metadata_extractor = MetadataExtractor()

        cet = pytz.timezone("Europe/Brussels")
        now = datetime.now(cet)
        self.session_id = now.strftime("%Y%m%d_%H%M%S")

        self.status_report = {
            "session_id": self.session_id,
            "start_time": now,
            "processed": [],
            "errors": []
        }

        # S3 setup
        self.s3_bucket = "cssf-crawl"
        self.s3_prefix = f"{self.session_id}/"
        self.s3 = boto3.client("s3")

        if self.mode == "sitemap":
            self.start_urls = ["https://www.cssf.lu/wp-sitemap.xml"]
        else:
            self.start_urls = ["https://www.cssf.lu/en/Document/circular-cssf-20-758/", "https://www.cssf.lu/en/2018/04/survey-related-to-the-fight-against-money-laundering-and-terrorist-financing-3/"]

    def start_requests(self):
        for url in self.start_urls:
            if "sitemap.xml" in url:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_main_sitemap,
                )
            else:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_document,
                    meta={
                        "super_category": "document" if "/Document/" in url else "post"
                    }
                )

    def parse_main_sitemap(self, response):
        sitemap_urls = SitemapHelper.parse_main_sitemap(response)
        for sitemap_url in sitemap_urls:
            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_document_sitemap,
            )

    def parse_document_sitemap(self, response):
        doc_urls = SitemapHelper.parse_document_sitemap(response)
        for url in doc_urls:
            if url not in self.visited:
                self.visited.add(url)
                super_category = "document" if "/Document/" in url else "post"
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_document,
                    meta={"super_category": super_category}
                )

    def parse_document(self, response):
        try:
            self.logger.info(f"Crawling: {response.url}")
            metadata = self.metadata_extractor.extract_metadata(response)
            if metadata:
                lang = response.url.split("/")[3] if len(response.url.split("/")) > 3 else "unknown"
                metadata.lang = lang
                metadata.super_category = response.meta.get("super_category", "unknown")

                if metadata.super_category == "post":
                    metadata.top_related = [response.url]

                url_path = urlparse(response.url).path.strip("/").replace("/", "-")
                s3_folder = f"{self.s3_prefix}{url_path}/"

                uploaded_files = []
                for url in metadata.top_related:
                    try:
                        filename = safe_filename_from_url(url)
                        s3_key = f"{s3_folder}{filename}"

                        r = requests.get(url)
                        r.raise_for_status()

                        content_type = r.headers.get("Content-Type", "").split(";")[0].strip()
                        if not content_type:
                            content_type = "application/octet-stream"

                        self.s3.put_object(
                            Bucket=self.s3_bucket,
                            Key=s3_key,
                            Body=r.content,
                            ContentType=content_type
                        )

                        uploaded_files.append({
                            "url": url,
                            "s3_uri": f"s3://{self.s3_bucket}/{s3_key}",
                            "content_type": content_type
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to upload {url} to S3: {e}")

                metadata.top_related = uploaded_files
                metadata_key = f"{s3_folder}metadata.json"
                self.s3.put_object(
                    Bucket=self.s3_bucket,
                    Key=metadata_key,
                    Body=json.dumps(metadata.__dict__, indent=2, default=str).encode("utf-8"),
                    ContentType="application/json"
                )

                self.logger.debug(f"Uploaded metadata to s3://{self.s3_bucket}/{metadata_key}")

                self.status_report["processed"].append(response.url)

                yield metadata.__dict__
        except Exception as e:
            self.logger.error(f"Error processing {response.url}: {str(e)}")
            self.status_report["errors"].append({"url": response.url, "error": str(e)})

    def closed(self, reason):
        cet = pytz.timezone("Europe/Brussels")
        now = datetime.now(cet)
        self.status_report["end_time"] = now
        status_key = f"{self.s3_prefix}status_report.json"
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key=status_key,
            Body=json.dumps(self.status_report, indent=2, default=str).encode("utf-8"),
            ContentType="application/json"
        )
        self.logger.info(f"Saved crawl status report to s3://{self.s3_bucket}/{status_key}")