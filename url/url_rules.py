import re
from urllib.parse import urlparse

import logging

logger = logging.getLogger(__name__)

class URLRules:
    def __init__(self):
        self.primary_domains = ["cssf.lu"]
        self.secondary_domains = [
            "eur-lex.europa.eu",
            "data.europa.eu",
            "ata.legilux.public.lu"
        ]
        self.exclude_url_patterns = [
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
        self.nested_only_patterns = [
            r"^https://www\.cssf\.lu/en/document",
            r"^https://www\.cssf\.lu/en/publication-data/",
            r"^https://www\.cssf\.lu/en/regulatory-framework/",
            r"^https://www\.cssf\.lu/en/?$"
        ]
        self.visited = set()

    def is_excluded(self, url):
        for pattern in self.exclude_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.debug(f"Excluded by pattern: {pattern} → {url}")
                return True
        return False

    def is_nested_only(self, url):
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in self.nested_only_patterns)

    def is_allowed_domain(self, url):
        domain = urlparse(url).netloc
        return any(allowed in domain for allowed in self.primary_domains + self.secondary_domains)

    def get_domain_type(self, url):
        domain = urlparse(url).netloc.lower().lstrip("www.")
        if any(domain == p or domain.endswith("." + p) for p in self.primary_domains):
            return "primary"
        if any(domain == s or domain.endswith("." + s) for s in self.secondary_domains):
            return "secondary"
        return "unknown"

    def is_visited(self, url):
        return url in self.visited

    def mark_visited(self, url):
        self.visited.add(url)

    def should_follow(self, url):
        if self.is_excluded(url):
            return False
        if self.is_visited(url):
            return False
        if not self.is_allowed_domain(url):
            return False
        #if not self.get_domain_type(url)  == "primary":
            #return False

        return True


