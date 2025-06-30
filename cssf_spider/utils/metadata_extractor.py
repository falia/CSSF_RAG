
import hashlib
import re
from datetime import datetime
from urllib.parse import urljoin
from typing import Optional, List, Dict
from items.metadata import DocumentMetadata


class MetadataExtractor:
    def extract_metadata(self, response) -> Optional[DocumentMetadata]:
        try:
            title = self.extract_title(response)
            subtitle = self.extract_subtitle(response)
            document_type = self.extract_document_type(response)
            document_number = self.extract_document_number(response.url, title)
            publication_date = self.extract_publication_date(response)
            update_date = self.extract_update_date(response)
            top_related = self.extract_related_documents(response)
            bottom_related = self.extract_bottom_related_documents(response)
            themes = self.extract_sidebar_themes(response)
            entities = self.extract_sidebar_entities(response)
            keywords = self.extract_keywords(response)
            content_hash = hashlib.sha256(response.text.encode('utf-8')).hexdigest()

            return DocumentMetadata(
                url=response.url,
                title=title,
                subtitle=subtitle,
                document_type=document_type,
                document_number=document_number,
                publication_date=publication_date,
                update_date=update_date,
                top_related=top_related,
                bottom_related=bottom_related,
                themes=themes,
                entities=entities,
                keywords=keywords,
                content_hash=content_hash,
                crawl_timestamp=datetime.now(),
                file_size=len(response.text.encode('utf-8')),
            )
        except Exception as e:
            response.meta.get('logger', None).error(f"Metadata extraction error: {e}")
            return None


    def extract_title(self, response) -> str:
        return response.css('h1.single-news__title::text').get(default='Unknown Document').strip()

    def extract_subtitle(self, response) -> str:
        return response.css('.single-news__subtitle p::text').get(default='').strip()

    def extract_document_type(self, response) -> str:
        return response.css('.main-category::text').get(default='Unknown Document Type').strip()

    def extract_document_number(self, url: str, title: str) -> str:
        match_url = re.search(r'cssf-(\d{2}-\d{3})', url)
        match_title = re.search(r'CSSF\s+(\d{2}/\d{3})', title)
        if match_url:
            return f"CSSF {match_url.group(1)}"
        elif match_title:
            return f"CSSF {match_title.group(1)}"
        return "Unknown"

    def extract_publication_date(self, response) -> Optional[str]:
        date_text = response.css('.single-news__date::text').get()
        if date_text:
            date_text = date_text.strip()

            # Normalize prefix in English, French, German
            date_text = re.sub(
                r'^(Published on|Publié le|Veröffentlicht am)\s+', '', date_text, flags=re.IGNORECASE
            )

            match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', date_text)
            return match.group(1) if match else None
        return None

    def extract_update_date(self, response) -> Optional[str]:
        date_text = response.css('.single-news__date--updated::text').get()
        if date_text:
            date_text = date_text.strip()

            # Normalize prefix in English, French, German
            date_text = re.sub(
                r'^(Updated on|Mis à jour le|Aktualisiert am)\s+', '', date_text, flags=re.IGNORECASE
            )

            match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', date_text)
            return match.group(1) if match else None
        return None

    def extract_related_documents(self, response) -> List[str]:
        """Extract all related document links from the structured top section"""
        related_docs = []
        section = response.css('.related-elements-container')
        anchors = section.css('a[href]')
        for anchor in anchors:
            href = anchor.css('::attr(href)').get()
            if not href:
                continue
            full_url = urljoin('https://www.cssf.lu', href) if href.startswith('/') else href
            related_docs.append(full_url)

        return list(set(related_docs))

    def extract_bottom_related_documents(self, response) -> List[str]:
        """Extract related documents from the bottom section"""
        links = response.css('.related-documents-container a[href*="/Document/"]::attr(href)').getall()
        return list({urljoin('https://www.cssf.lu', href) for href in links if '/Document/' in href})


    def extract_keywords(self, response) -> List[str]:
            return [k.strip() for k in response.css('.keywords a::text').getall() if k.strip()]

    
    def extract_sidebar_themes(self, response) -> List[str]:
        """Extract theme names from the sidebar"""
        return [
            a.css("::text").get().strip()
            for a in response.css(".themes-list a")
            if a.css("::text").get()
        ]

    def extract_sidebar_entities(self, response) -> List[str]:
        """Extract entity names from the sidebar"""
        return [
            a.css("::text").get().strip()
            for a in response.css(".entities-list a")
            if a.css("::text").get()
        ]