import requests
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Tuple
from .config import settings
 
 
def _auth():
    return (settings.CONFLUENCE_EMAIL, settings.CONFLUENCE_API_TOKEN)
 
 
def fetch_pages_in_space(limit: int = 25) -> List[Dict[str, Any]]:
    url = f"{settings.CONFLUENCE_BASE_URL}/rest/api/content"
    params = {
        "spaceKey": settings.CONFLUENCE_SPACE_KEY,
        "type": "page",
        "limit": limit,
        "expand": "body.storage,version,metadata.labels",
    }
    r = requests.get(url, params=params, auth=_auth(), timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])
 
 
def storage_html_to_text(storage_html: str) -> str:
    soup = BeautifulSoup(storage_html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])
 
 
def extract_tables(storage_html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(storage_html, "lxml")
    tables = []
    for t_i, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append({"table_index": t_i, "rows": rows})
    return tables
 
 
def table_to_text(table: Dict[str, Any]) -> str:
    return "TABLE:\n" + "\n".join(" | ".join(r) for r in table["rows"])
 
 
def page_to_doc_parts(page: Dict[str, Any]) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    page_id = str(page["id"])
    title = page.get("title", "")
    storage_html = page.get("body", {}).get("storage", {}).get("value", "") or ""
    updated_at = page.get("version", {}).get("when", "")
    url = f"{settings.CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={page_id}"
 
    labels = []
    for lab in page.get("metadata", {}).get("labels", {}).get("results", []):
        labels.append(lab.get("name"))
 
    meta = {
        "page_id": page_id,
        "title": title,
        "space_key": settings.CONFLUENCE_SPACE_KEY,
        "updated_at": updated_at,
        "labels": labels,
        "url": url,
    }
    return meta, storage_html_to_text(storage_html), extract_tables(storage_html)