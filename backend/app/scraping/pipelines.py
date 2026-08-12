"""
Scrapy pipelines for NBA player headshot scraping.

The HeadshotPipeline:
  1. Upgrades each headshot URL from 260x190 → 1040x760 resolution.
  2. Collects all items into an in-memory dict.
  3. On spider close, writes the final JSON artifact to
     ``backend/data/nba_player_headshots.json`` and logs match coverage
     against the dataset CSV (``output_players/players_with_archetypes.csv``).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from itemadapter import ItemAdapter

from .url_allowlist import is_allowed_headshot_url

logger = logging.getLogger(__name__)

# ── Paths ──
# backend/app/scraping/pipelines.py → workspace root is 3 parents up
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_DIR = _WORKSPACE_ROOT / "backend" / "data"
_OUTPUT_FILE = _OUTPUT_DIR / "nba_player_headshots.json"
_DATASET_CSV = _WORKSPACE_ROOT / "output_players" / "players_with_archetypes.csv"

# Size upgrade: 260x190 thumbnail → 1040x760 full resolution
_SIZE_REPLACEMENT = ("260x190", "1040x760")


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching.

    - Lowercase
    - Strip diacritics (é→e, č→c, ş→s, etc.) via NFD decomposition
    - Strip suffixes: Jr., Sr., III, IV, II
    - Collapse whitespace
    - Remove periods and apostrophes
    """
    import unicodedata

    # NFD decomposition: split base chars from combining marks, then drop marks
    normalized = unicodedata.normalize("NFD", name)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    normalized = normalized.lower().strip()
    # Remove generational suffixes
    normalized = re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", "", normalized)
    # Remove periods and apostrophes
    normalized = normalized.replace(".", "").replace("'", "")
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


class HeadshotPipeline:
    """Collect, upgrade, and persist headshot URLs."""

    def __init__(self) -> None:
        self._headshots: dict[str, str] = {}  # normalized_name → url
        self._raw_names: dict[str, str] = {}  # normalized_name → original_name

    def process_item(self, item: Any, spider: Any = None) -> Any:
        adapter = ItemAdapter(item)
        name: str = adapter.get("name", "")
        url: str = adapter.get("headshot_url", "")

        if not name or not url:
            return item

        # Defense-in-depth: only accept https URLs on the NBA CDN allowlist.
        # Rejects javascript:/data: schemes, non-CDN hosts, and tampered URLs.
        if not is_allowed_headshot_url(url):
            logger.warning(
                "Rejecting headshot URL for %r — not on allowlist: %s", name, url
            )
            return item

        # Upgrade resolution: 260x190 → 1040x760
        if _SIZE_REPLACEMENT[0] in url:
            url = url.replace(_SIZE_REPLACEMENT[0], _SIZE_REPLACEMENT[1])

        normalized = _normalize_name(name)

        # Keep the first occurrence (avoid overwriting with duplicates)
        if normalized not in self._headshots:
            self._headshots[normalized] = url
            self._raw_names[normalized] = name

        return item

    def close_spider(self, spider: Any = None) -> None:
        """Write the JSON artifact and log match coverage."""
        if not self._headshots:
            logger.warning("No headshots collected — skipping JSON write")
            return

        # ── Write the artifact ──
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Write keyed by original player name (not normalized) for readability
        output: dict[str, str] = {
            self._raw_names[norm]: url
            for norm, url in sorted(self._headshots.items())
        }

        with _OUTPUT_FILE.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False, sort_keys=True)

        logger.info("Wrote %d headshots to %s", len(output), _OUTPUT_FILE)

        # ── Log match coverage against the dataset CSV ──
        self._log_match_coverage()

    def _log_match_coverage(self) -> None:
        """Compare the scraped names against the dataset CSV player names."""
        if not _DATASET_CSV.exists():
            logger.warning("Dataset CSV not found at %s — skipping coverage check", _DATASET_CSV)
            return

        dataset_names: list[str] = []
        with _DATASET_CSV.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                player_name = row.get("player", "")
                if player_name:
                    dataset_names.append(player_name)

        matched = 0
        unmatched: list[str] = []

        for name in dataset_names:
            if _normalize_name(name) in self._headshots:
                matched += 1
            else:
                unmatched.append(name)

        total = len(dataset_names)
        coverage = (matched / total * 100) if total else 0.0

        logger.info(
            "Headshot match coverage: %d/%d (%.1f%%)",
            matched,
            total,
            coverage,
        )

        if unmatched:
            logger.warning(
                "Unmatched players (%d): %s",
                len(unmatched),
                ", ".join(unmatched[:50]),
            )
