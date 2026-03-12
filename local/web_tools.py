"""
Rio Local — Web Tools (E3)

Provides web_search, web_fetch, and web_cache_get for quick web research
without needing to open a browser.

- web_search: DuckDuckGo Instant Answer API (no API key needed)
- web_fetch: HTTP GET with HTML→text extraction + SSRF protection
- web_cache_get: LRU cache with 1-hour TTL
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# SSRF protection: block private/reserved IPs
# ---------------------------------------------------------------------------
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_url(url: str) -> bool:
    """Check if a URL resolves to a private/reserved IP (SSRF protection)."""
    try:
        hostname = urlparse(url).hostname
        if not hostname:
            return True
        import socket
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        for family, type_, proto, canonname, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _BLOCKED_NETWORKS:
                if ip in network:
                    return True
    except Exception:
        return True  # Block on resolution failure
    return False


# ---------------------------------------------------------------------------
# Simple response cache
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 3600  # 1 hour


def _cache_get(key: str) -> Any | None:
    """Get from cache if not expired."""
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _CACHE_TTL:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    """Store in cache. Evict oldest if cache > 200 entries."""
    if len(_cache) > 200:
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest_key, None)
    _cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Strip HTML tags and decode entities, returning plain text."""
    try:
        from html.parser import HTMLParser
        from html import unescape

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False
                if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
                    self._parts.append("\n")

            def handle_data(self, data):
                if not self._skip:
                    self._parts.append(data)

        stripper = _Stripper()
        stripper.feed(unescape(html))
        text = "".join(stripper._parts)
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        # Fallback: regex strip
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo (no API key required).

    Returns top results with title, url, and snippet.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Empty query"}

    import urllib.request
    import urllib.parse
    import json as _json

    cache_key = f"search:{query}:{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        # DuckDuckGo Instant Answer API
        params = urllib.parse.urlencode({
            "q": query.strip(),
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Rio-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Answer"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["Abstract"][:300],
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")[:300],
                })

        result = {"success": True, "query": query, "results": results[:max_results]}
        _cache_set(cache_key, result)
        return result

    except Exception as exc:
        log.warning("web_search.error", query=query[:60], error=str(exc))
        return {"success": False, "error": f"Search failed: {exc}"}


def web_fetch(url: str, max_chars: int = 8000) -> dict:
    """Fetch a web page and return its text content.

    Includes SSRF protection against private/reserved IPs.
    HTML is converted to plain text automatically.
    """
    if not url or not url.strip():
        return {"success": False, "error": "Empty URL"}

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # SSRF protection
    if _is_private_url(url):
        return {"success": False, "error": "URL resolves to a private/reserved IP (blocked)"}

    import urllib.request

    cache_key = f"fetch:{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Rio-Agent/1.0 (Desktop Assistant)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(500_000)  # Cap at 500KB
            encoding = "utf-8"
            if "charset=" in content_type:
                encoding = content_type.split("charset=")[-1].split(";")[0].strip()
            text = raw.decode(encoding, errors="replace")

        # Convert HTML to plain text
        if "html" in content_type.lower():
            text = _html_to_text(text)

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"

        result = {"success": True, "url": url, "content": text, "length": len(text)}
        _cache_set(cache_key, result)
        return result

    except Exception as exc:
        log.warning("web_fetch.error", url=url[:80], error=str(exc))
        return {"success": False, "error": f"Fetch failed: {exc}"}


def web_cache_get(url: str) -> dict:
    """Get a cached web page. Returns cached content if available,
    otherwise fetches and caches it."""
    cache_key = f"fetch:{url.strip()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached
    result = web_fetch(url)
    if result.get("success"):
        result["cache_hit"] = False
    return result
