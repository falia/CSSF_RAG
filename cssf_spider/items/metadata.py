
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class DocumentMetadata:
    url: str
    title: str
    subtitle: str
    document_type: str
    document_number: str
    publication_date: Optional[str]
    update_date: Optional[str]
    top_related: List[str]
    bottom_related: List[str]
    themes: List[str]
    entities: List[str]
    keywords: List[str]
    content_hash: str
    crawl_timestamp: datetime
    file_size: int
    lang: str
    super_category: str

