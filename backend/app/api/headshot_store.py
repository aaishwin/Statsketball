"""
Headshot Data Store
===================
Read-only store for the NBA player headshot URL mapping.

Loads ``backend/data/nba_player_headshots.json`` (produced by the
Scrapy-Playwright spider) lazily on first access and caches it for the
process lifetime. The file only changes when the spider is re-run
manually, which requires an API restart anyway.

The store performs name normalization (diacritic stripping, suffix
removal, case-folding) so that lookups succeed regardless of minor
spelling differences between the dataset and NBA.com.

Mirrors the pattern in ``archetype_store.py``: lazy singleton, thread-safe
initialization, workspace-root-relative path resolution.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..scraping.url_allowlist import is_allowed_headshot_url

logger = logging.getLogger(__name__)

# backend/app/api/headshot_store.py -> workspace root is 3 parents up
_WORKSPACE_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_JSON_NAME: Final[str] = "nba_player_headshots.json"
_JSON_DIR: Final[Path] = _WORKSPACE_ROOT / "backend" / "data"


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching.

    - NFD decomposition + strip combining marks (é→e, č→c, ş→s)
    - Lowercase
    - Strip generational suffixes (Jr., Sr., III, IV, II)
    - Remove periods and apostrophes
    - Collapse whitespace
    """
    normalized = unicodedata.normalize("NFD", name)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", "", normalized)
    normalized = normalized.replace(".", "").replace("'", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


@dataclass(slots=True, frozen=True)
class HeadshotStore:
    """Immutable loaded headshot mapping: normalized_name → CDN URL."""

    by_normalized: dict[str, str] = field(default_factory=dict)
    by_original: dict[str, str] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return bool(self.by_normalized)

    def get_url(self, name: str) -> str | None:
        """Look up a headshot URL by player name.

        Returns the full CDN URL (1040x760) or ``None`` if no match.
        """
        key = _normalize_name(name)
        return self.by_normalized.get(key)

    def get_all(self) -> dict[str, str]:
        """Return the full mapping keyed by original player names."""
        return dict(self.by_original)


_store: HeadshotStore | None = None
_lock = threading.Lock()


def _json_path() -> Path:
    override: str | None = os.environ.get("HEADSHOT_JSON_PATH")
    if override:
        return Path(override)
    return _JSON_DIR / _JSON_NAME


def _load_store() -> HeadshotStore:
    """Load the headshot JSON and build the normalized lookup index."""
    path = _json_path()

    if not path.exists():
        logger.warning(
            "Headshot mapping not found at %s — headshots will be unavailable. "
            "Run `python run_scrape.py` to generate it.",
            path,
        )
        return HeadshotStore()

    with path.open(encoding="utf-8") as fh:
        raw: dict[str, str] = json.load(fh)

    by_normalized: dict[str, str] = {}
    skipped = 0
    for name, url in raw.items():
        # Defense-in-depth: validate again at load time. This protects
        # against artifact tampering (not just bad scrapes) — a tampered
        # JSON file with javascript:/data: or non-CDN URLs is skipped here.
        if not is_allowed_headshot_url(url):
            logger.warning(
                "Skipping headshot for %r — URL not on allowlist: %s", name, url
            )
            skipped += 1
            continue
        key = _normalize_name(name)
        if key not in by_normalized:
            by_normalized[key] = url

    logger.info(
        "Loaded %d headshot mappings from %s (%d skipped by allowlist)",
        len(by_normalized), path, skipped,
    )
    return HeadshotStore(by_normalized=by_normalized, by_original={
        name: url for name, url in raw.items() if is_allowed_headshot_url(url)
    })


def get_store() -> HeadshotStore:
    """Return the singleton HeadshotStore, loading on first access."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = _load_store()
    return _store
