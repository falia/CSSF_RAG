
import os
import requests
from urllib.parse import urlparse
from scrapy.http import HtmlResponse
from typing import List, Tuple
import base64


def safe_filename_from_url(url: str) -> str:
    """Encodes the full URL as a filename-safe base64 string with .html/.pdf/etc extension preserved if possible."""
    parsed = urlparse(url)
    extension = os.path.splitext(parsed.path)[1] or ".html"
    b64 = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"{b64}{extension}"

def is_downloadable_document(url: str) -> bool:
    try:
        if url.lower().endswith('.pdf'):
            return True
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get('Content-Type', '').lower()
        return 'text/html' in content_type
    except Exception:
        return False


def download_related_documents(urls: List[str], save_dir: str) -> List[Tuple[str, str]]:
    downloaded_files = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for url in urls:
        try:
            file_name = safe_filename_from_url(url)
            if not file_name:
                continue
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append((file_path, content_type))
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return downloaded_files

