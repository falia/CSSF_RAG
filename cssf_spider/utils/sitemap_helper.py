import xml.etree.ElementTree as ET

""" TODO add pages from sitemap - requires separate meta data extractor """
""" TODO sometimes we have same document with three different urls but same content, to investigate if we remove duplicates """
class SitemapHelper:
    @staticmethod
    def parse_main_sitemap(response) -> list:
        """Extract all 'document' and 'post' sitemaps, regardless of language"""
        sitemap_urls = []
        try:
            root = ET.fromstring(response.text)
            ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            sitemaps = root.findall('.//sitemap:loc', ns)
            for sitemap in sitemaps:
                text = sitemap.text or ''
                if (
                    'en/wp-sitemap-posts-document-' in text
                    or 'en/wp-sitemap-posts-post-' in text
                ):
                    sitemap_urls.append(text)
        except ET.ParseError as e:
            response.meta['logger'].error(f"Main sitemap XML parse error: {e}")
        return sitemap_urls

    @staticmethod
    def parse_document_sitemap(response) -> list:
        """Extract all URLs listed in a document or post sitemap"""
        urls = []
        try:
            root = ET.fromstring(response.text)
            ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            url_elements = root.findall('.//sitemap:url', ns)
            for elem in url_elements:
                loc = elem.find('sitemap:loc', ns)
                if loc is not None and loc.text:
                    urls.append(loc.text)
        except ET.ParseError as e:
            response.meta['logger'].error(f"Document sitemap XML parse error: {e}")
        return urls
