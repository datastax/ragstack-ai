from typing import TYPE_CHECKING, Iterable, Iterator, Sequence, Set, Union
from urllib.parse import urldefrag, urljoin, urlparse
from ragstack_knowledge_store.edge_extractor import EdgeExtractor, IncomingLinkTag, LinkTag, OutgoingLinkTag, get_link_tags
from langchain_core.documents import Document
from ._utils import strict_zip


if TYPE_CHECKING:
   from bs4 import BeautifulSoup

def _parse_url(link,
              page_url,
              drop_fragments: bool = True):
  href = link.get('href')
  if href is None:
    return None
  url = urlparse(href)
  if url.scheme not in ["http", "https", ""]:
    return None

  # Join the HREF with the page_url to convert relative paths to absolute.
  url = urljoin(page_url, href)

  # Fragments would be useful if we chunked a page based on section.
  # Then, each chunk would have a different URL based on the fragment.
  # Since we aren't doing that yet, they just "break" links. So, drop
  # the fragment.
  if drop_fragments:
    return urldefrag(url).url
  else:
     return url

def _parse_hrefs(soup: "BeautifulSoup",
                 url: str,
                 drop_fragments: bool = True) -> Set[str]:
  links = soup.find_all('a')
  links = {_parse_url(link, page_url=url, drop_fragments=drop_fragments) for link in links}

  # Remove entries for any 'a' tag that failed to parse (didn't have href,
  # or invalid domain, etc.)
  links.discard(None)

  # Remove self links.
  links.discard(url)

  return links

class HtmlLinkEdgeExtractor(EdgeExtractor[Union[str, "BeautifulSoup"]]):
    def __init__(self,
                 url_field: str = "source",
                 *,
                 kind: str = "hyperlink",
                 drop_fragments: bool = True):
        """Extract hyperlinks from HTML content.

        Expects the `page_content` to be HTML.

        Args:
            url_field: Name of the metadata field containing the URL
                of the content. Defaults to "source".
            kind: The kind of edge to extract. Defaults to "hyperlink".
            drop_fragmets: Whether fragments in URLs and links shoud be
                dropped. Defaults to `True`.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "BeautifulSoup4 is required for HtmlLinkEdgeExtractor. "
                "Please install it with `pip install beautifulsoup4`."
            )

        self.url_field = url_field
        self._kind = kind
        self.drop_fragments = drop_fragments

    def extract_one(
          self,
          document: Document,
          input: Union[str, "BeautifulSoup"],
    ):
        if isinstance(input, str):
            from bs4 import BeautifulSoup
            input = BeautifulSoup(input, "html.parser")

        url = document.metadata[self.url_field]
        if self.drop_fragments:
            url = urldefrag(url).url

        hrefs = _parse_hrefs(input, url, self.drop_fragments)

        link_tags = get_link_tags(document)
        link_tags.add(IncomingLinkTag(kind=self._kind, tag=url))
        for url in hrefs:
            link_tags.add(OutgoingLinkTag(kind=self._kind, tag=url))
