"""
Headshot URL allowlist validation.

Defense-in-depth for scraped headshot URLs: NBA.com headshots should only
ever come from a small set of known CDN hosts over HTTPS. URLs are taken
from scraped page DOM and flow into a JSON artifact served to all frontend
clients, where they're rendered as <img src>. If NBA.com is compromised,
serves injected ad content, or the JSON file is tampered with, the app
could otherwise distribute attacker-controlled URLs (tracking pixels,
mixed content, javascript:/data: schemes).

Validated at TWO layers (per SECURITY_AUDIT_PLAN.md M4):
  1. The Scrapy pipeline (pipelines.py) — rejects bad URLs at scrape time.
  2. The headshot store (headshot_store.py) — rejects bad URLs at load time,
     protecting against artifact tampering, not just bad scrapes.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: Allowed hostnames for NBA headshot CDN URLs.
#: NBA.com serves player headshots from these CDN hosts. Any URL whose
#: hostname is not in this set is rejected.
_ALLOWED_HEADSHOT_HOSTS: frozenset[str] = frozenset({
    "cdn.nba.com",
    "ak-static.cms.nba.com",
})


def is_allowed_headshot_url(url: str) -> bool:
    """Return True iff ``url`` is an https URL on an allowlisted CDN host.

    Rules:
    - Scheme must be exactly ``https`` (no http, javascript, data, etc.).
    - Hostname must be in ``_ALLOWED_HEADSHOT_HOSTS``.
    - Path must be non-empty.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    if parsed.scheme != "https":
        return False
    if (parsed.hostname or "").lower() not in _ALLOWED_HEADSHOT_HOSTS:
        return False
    if not parsed.path:
        return False
    return True
