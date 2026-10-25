"""HTML -> plain text helper for CRAG's raw page_result field."""
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MAX_CHARS = 4000  # keep passages short enough for TF-IDF + prompt budgets


def html_to_text(html: str, max_chars: int = MAX_CHARS) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    return text[:max_chars]
